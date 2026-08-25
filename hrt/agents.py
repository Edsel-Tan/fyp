#!/usr/bin/env python3
"""PPO and DDPG implementations used for both the HRT controllers and the
standalone baselines, so that any performance gap is attributable to the
hierarchy rather than to differing library internals.

PPO covers two action heads:
  "multi"    factorised categorical over N x 3  -> HRT high-level controller
  "gaussian" diagonal Gaussian over [-1,1]^N    -> standalone PPO baseline
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


# --------------------------------------------------------------------- PPO
class PPO:
    def __init__(self, obs_dim, n_assets, head="multi", hidden=256, lr=3e-4,
                 clip=0.2, epochs=10, gamma=0.99, lam=0.95, ent_coef=0.01,
                 vf_coef=0.5, minibatch=256, device="cuda", factored=True):
        """factored=True treats the N action dimensions as N near-independent
        decisions: the clipped objective and the entropy bonus are averaged over
        dimensions rather than summed.

        Summing is what a stock PPO does, and it is fine at the handful of action
        dimensions such implementations are usually run on. At N=370 it is not:
        the entropy bonus becomes ent_coef*N*ln(3) ~ 4.1 against a policy-gradient
        term of order 0.01, so the policy is driven to uniform random and stays
        there, and the joint importance ratio -- a product of 370 per-dimension
        ratios -- is so volatile that the clip branch is taken almost always.
        """
        self.factored = factored
        self.n, self.head, self.dev = n_assets, head, device
        self.clip, self.epochs, self.gamma, self.lam = clip, epochs, gamma, lam
        self.ent_coef, self.vf_coef, self.minibatch = ent_coef, vf_coef, minibatch
        out = n_assets * 3 if head == "multi" else n_assets
        self.pi = mlp([obs_dim, hidden, hidden, out]).to(device)
        self.v = mlp([obs_dim, hidden, hidden, 1]).to(device)
        params = list(self.pi.parameters()) + list(self.v.parameters())
        if head == "gaussian":
            self.log_std = nn.Parameter(torch.full((n_assets,), -0.5, device=device))
            params.append(self.log_std)
        self.opt = torch.optim.Adam(params, lr=lr)

    def _dist(self, obs):
        o = self.pi(obs)
        if self.head == "multi":
            return torch.distributions.Categorical(logits=o.view(*o.shape[:-1], self.n, 3))
        return torch.distributions.Normal(torch.tanh(o), self.log_std.exp())

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Returns (env action, raw action, joint log-prob, value)."""
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.dev).unsqueeze(0)
        d = self._dist(o)
        if self.head == "multi":
            a = d.logits.argmax(-1) if deterministic else d.sample()
            logp = d.log_prob(a).sum(-1)
            env_a = (a.squeeze(0).cpu().numpy() - 1)          # {0,1,2} -> {-1,0,1}
        else:
            a = d.mean if deterministic else d.sample()
            logp = d.log_prob(a).sum(-1)
            env_a = np.clip(a.squeeze(0).cpu().numpy(), -1, 1)
        return env_a, a.squeeze(0).cpu().numpy(), float(logp), float(self.v(o))

    def update(self, obs, acts, logps, rews, dones, vals, last_val, adv_dim=None):
        """adv_dim [T,N]: an immediate, per-dimension advantage added to the
        temporal one. The HLC reward is a *sum of per-stock* alignment terms, and
        each term depends only on that stock's own action, so it carries direct
        per-dimension credit -- collapsing it to a scalar before the update throws
        that away and leaves 370 decisions sharing one learning signal."""
        dev = self.dev
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=dev)
        acts = torch.as_tensor(np.asarray(acts), device=dev)
        if self.head == "multi":
            acts = acts.long()
        else:
            acts = acts.float()
        logps = torch.as_tensor(np.asarray(logps), dtype=torch.float32, device=dev)
        vals = np.asarray(vals + [last_val], np.float64)
        rews, dones = np.asarray(rews, np.float64), np.asarray(dones, np.float64)

        adv, gae = np.zeros_like(rews), 0.0
        for t in reversed(range(len(rews))):
            nonterm = 1.0 - dones[t]
            delta = rews[t] + self.gamma * vals[t + 1] * nonterm - vals[t]
            gae = delta + self.gamma * self.lam * nonterm * gae
            adv[t] = gae
        ret = adv + vals[:-1]
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        if adv_dim is not None:
            ad = np.asarray(adv_dim, np.float64)
            ad = (ad - ad.mean()) / (ad.std() + 1e-8)
            adv_full = adv_n[:, None] + ad                     # [T, N]
        else:
            adv_full = adv_n[:, None]
        adv_t = torch.as_tensor(adv_full, dtype=torch.float32, device=dev)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=dev)

        with torch.no_grad():
            old_lp_d = self._dist(obs).log_prob(acts) if self.factored else None

        idx = np.arange(len(rews))
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), self.minibatch):
                j = torch.as_tensor(idx[s:s + self.minibatch], device=dev)
                d = self._dist(obs[j])
                if self.factored:
                    # per-dimension ratio and objective, averaged over dimensions
                    lp_d = d.log_prob(acts[j])                    # [B, N]
                    ratio = (lp_d - old_lp_d[j]).exp()
                    adv_b = adv_t[j]                              # [B, N] or [B, 1]
                    l1 = ratio * adv_b
                    l2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_b
                    pg = -torch.min(l1, l2).mean()
                    ent = d.entropy().mean()
                else:
                    lp = d.log_prob(acts[j]).sum(-1)
                    ratio = (lp - logps[j]).exp()
                    a_s = adv_t[j].squeeze(-1) if adv_t.shape[-1] == 1 else adv_t[j].mean(-1)
                    l1 = ratio * a_s
                    l2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a_s
                    pg = -torch.min(l1, l2).mean()
                    ent = d.entropy().sum(-1).mean()
                vl = F.mse_loss(self.v(obs[j]).squeeze(-1), ret_t[j])
                loss = pg + self.vf_coef * vl - self.ent_coef * ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
                self.opt.step()


# -------------------------------------------------------------------- DDPG
class Replay:
    def __init__(self, size, obs_dim, act_dim):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.d = np.zeros(size, np.float32)
        self.size, self.ptr, self.full = size, 0, False

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.size
        self.full |= self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n, dev):
        j = np.random.randint(0, len(self), n)
        t = lambda x: torch.as_tensor(x[j], device=dev)
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


class DDPG:
    def __init__(self, obs_dim, act_dim, hidden=256, actor_lr=1e-3, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, buffer=200_000, batch=256, device="cuda"):
        self.dev, self.gamma, self.tau, self.batch = device, gamma, tau, batch
        self.actor = mlp([obs_dim, hidden, hidden, act_dim], out_act=nn.Tanh).to(device)
        self.critic = mlp([obs_dim + act_dim, hidden, hidden, 1]).to(device)
        self.actor_t = mlp([obs_dim, hidden, hidden, act_dim], out_act=nn.Tanh).to(device)
        self.critic_t = mlp([obs_dim + act_dim, hidden, hidden, 1]).to(device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buf = Replay(buffer, obs_dim, act_dim)
        self._src_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self._tgt_params = list(self.actor_t.parameters()) + list(self.critic_t.parameters())

    @torch.no_grad()
    def act(self, obs, noise=0.0):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.dev).unsqueeze(0)
        a = self.actor(o).squeeze(0).cpu().numpy()
        if noise > 0:
            a = a + np.random.normal(0, noise, a.shape)
        return np.clip(a, -1, 1)

    def update(self, steps=1):
        if len(self.buf) < 2 * self.batch:
            return
        for _ in range(steps):
            o, a, r, o2, d = self.buf.sample(self.batch, self.dev)
            with torch.no_grad():
                y = r + self.gamma * (1 - d) * self.critic_t(torch.cat([o2, self.actor_t(o2)], -1)).squeeze(-1)
            q = self.critic(torch.cat([o, a], -1)).squeeze(-1)
            lc = F.mse_loss(q, y)
            self.opt_c.zero_grad(); lc.backward(); self.opt_c.step()

            la = -self.critic(torch.cat([o, self.actor(o)], -1)).mean()
            self.opt_a.zero_grad(); la.backward(); self.opt_a.step()

            with torch.no_grad():
                torch._foreach_lerp_(self._tgt_params, self._src_params, self.tau)
