#!/usr/bin/env python3
"""Train HRT and the standalone DRL baselines, then evaluate on 2021 / 2022.

HRT follows the paper's Phased Alternating Training: each iteration collects a
rollout in which the PPO high-level controller picks a direction per stock and
the DDPG low-level controller sizes the trade; the HLC reward blends a
directional-alignment term with the LLC's realised portfolio-value change under
an exponentially decaying weight, and both controllers are updated before the
next rollout.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import TradingEnv, metrics
from agents import PPO, DDPG

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

PERIODS = {"train": ("2015-01-01", "2019-12-31"), "valid": ("2020-01-01", "2020-12-31"),
           "test2021": ("2021-01-01", "2021-12-31"), "test2022": ("2022-01-01", "2022-12-31")}


def build_data(panel, frfile, signal_mode):
    """signal_mode:
      causal  -- forecast formed at close of t, targeting open_{t+1}->open_{t+2}
      paper   -- the paper's literal timing, open_t->open_{t+1} (leaks the day-t close)
      shuffle -- causal forecast with its dates permuted: same distribution, no content
      none    -- no signal at all (zeros): isolates the hierarchy itself
    """
    z = np.load(panel, allow_pickle=True)
    dates = z["dates"].astype(str)
    op = z["open"].astype(np.float64)
    f = np.load(frfile, allow_pickle=True)
    assert (f["dates"].astype(str) == dates).all()
    fr = np.nan_to_num(f["fr"].astype(np.float64))
    if signal_mode == "none":
        fr = np.zeros_like(fr)
    if signal_mode in ("causal", "shuffle", "none"):
        # fr[t] was formed at the close of day t and targets open_{t+1}->open_{t+2},
        # so the agent trading at the open of day t+1 reads fr[t]: shift by one day.
        fr = np.vstack([np.zeros((1, fr.shape[1])), fr[:-1]])
    if signal_mode == "shuffle":
        # destroy the time alignment but keep every day's cross-section intact
        rng = np.random.default_rng(12345)
        fr = fr[rng.permutation(len(fr))]
    # standardise the signal cross-sectionally so the HLC sees a stable scale
    mu, sd = fr.mean(1, keepdims=True), fr.std(1, keepdims=True)
    fr = (fr - mu) / np.where(sd < 1e-12, 1.0, sd)
    fr = np.nan_to_num(fr)

    out = {}
    for k, (lo, hi) in PERIODS.items():
        i = np.where((dates >= lo) & (dates <= hi))[0]
        out[k] = {"prices": op[i], "fr": fr[i], "dates": dates[i]}
    out["n_assets"] = op.shape[1]
    return out


def evaluate(step_fn, data, key, seed=0):
    e = TradingEnv(data[key]["prices"], data[key]["fr"], seed=seed)
    done = False
    while not done:
        done = step_fn(e)
    m = metrics(e.values)
    m["turnover"] = float(np.mean(e.turnover))
    m["cost_paid"] = float(e.cost_paid)
    m["n_trades"] = int(e.trades)
    m["values"] = [float(v) for v in e.values]

    # activity structure, for the paper's inertia / diversification claims
    tl = np.abs(np.array(e.trade_log))                    # [steps, N] share counts
    traded_any = (tl > 0)
    per_name = tl.sum(0)
    share = per_name / max(per_name.sum(), 1)
    m["active_names_per_day"] = float(traded_any.sum(1).mean())
    m["names_ever_traded"] = int((per_name > 0).sum())
    m["trade_concentration_hhi"] = float((share ** 2).sum())
    m["trade_share_top10pct"] = float(np.sort(share)[::-1][:max(1, len(share) // 10)].sum())
    m["_trade_matrix"] = np.array(e.trade_log)
    return m


# ------------------------------------------------------------------- HRT
def make_hrt_stepper(hlc, llc, deterministic=True):
    def step(e):
        ah, _, _, _ = hlc.act(e.hlc_state(), deterministic=deterministic)
        sl = np.concatenate([e.state(), ah.astype(np.float32)])
        al = llc.act(sl, noise=0.0)
        _, _, done, _ = e.step_hier(ah, al)
        return done
    return step


def train_hrt(data, args, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    N = data["n_assets"]
    dev = args.device
    hlc = PPO(N, N, head="multi", lr=args.ppo_lr, device=dev)
    llc = DDPG(3 * N + 1, N, actor_lr=args.ddpg_lr, critic_lr=args.ddpg_lr,
               batch=args.llc_batch, buffer=args.buffer, device=dev)

    e = TradingEnv(data["train"]["prices"], data["train"]["fr"], seed=seed)
    obs_h = e.hlc_state()
    gstep, best, best_state, hist = 0, -9e9, None, []
    episode = 0
    t0 = time.time()

    while gstep < args.timesteps:
        O, A, LP, R, D, V, AD = [], [], [], [], [], [], []
        for _ in range(args.rollout):
            ah, raw, logp, val = hlc.act(obs_h)
            sl = np.concatenate([e.state(), ah.astype(np.float32)])
            noise = args.noise * max(0.0, 1.0 - gstep / (0.6 * args.timesteps))
            al = llc.act(sl, noise=noise if gstep >= args.warmup else 1.0)
            if gstep < args.warmup:
                al = np.random.uniform(-1, 1, N)
            _, r_l, done, info = e.step_hier(ah, al)
            sl2 = np.concatenate([e.state(), ah.astype(np.float32)])
            llc.buf.add(sl, al, r_l, sl2, float(done))

            # The paper gives alpha_t = alpha_0 * exp(-lambda*t) with lambda=1e-3 but
            # never says what t counts. Per env step, alpha is spent within ~3k steps
            # (2 episodes) and the alignment term is effectively a brief warm-up; per
            # episode, it decays across the whole 400-episode run. Both are tested.
            t_decay = episode if args.alpha_unit == "episode" else gstep
            alpha = args.alpha0 * np.exp(-args.lam * t_decay)
            r_h = alpha * float(info["align"].mean()) + (1 - alpha) * r_l

            O.append(obs_h); A.append(raw); LP.append(logp)
            R.append(r_h); D.append(float(done)); V.append(val)
            AD.append(alpha * info["align"])      # per-stock credit, faithful to r_h
            gstep += 1
            if done:
                episode += 1
                e.reset()
            obs_h = e.hlc_state()

        _, _, _, last_val = hlc.act(obs_h)
        hlc.update(O, A, LP, R, D, V, last_val, adv_dim=np.array(AD))
        if gstep >= args.warmup:
            llc.update(steps=max(1, args.rollout // args.update_every))

        if gstep % args.eval_every < args.rollout:
            m = evaluate(make_hrt_stepper(hlc, llc), data, "valid", seed)
            hist.append({"step": gstep, "valid_sharpe": m["sharpe"], "valid_cum": m["cum_return"]})
            if m["sharpe"] > best:
                best = m["sharpe"]
                best_state = ({k: v.detach().clone() for k, v in hlc.pi.state_dict().items()},
                              {k: v.detach().clone() for k, v in llc.actor.state_dict().items()})
            if args.verbose:
                print(f"  [{seed}] step {gstep:6d} valid sharpe {m['sharpe']:+.3f} "
                      f"cum {m['cum_return']:+.3f} best {best:+.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    if best_state:
        hlc.pi.load_state_dict(best_state[0]); llc.actor.load_state_dict(best_state[1])
    return make_hrt_stepper(hlc, llc), hist, best


# ------------------------------------------------------- standalone agents
def make_flat_stepper(agent, kind):
    def step(e):
        s = e.state()
        a = agent.act(s, deterministic=True)[0] if kind == "ppo" else agent.act(s, noise=0.0)
        _, _, done, _ = e.step(a)
        return done
    return step


def train_flat(data, args, seed, kind):
    torch.manual_seed(seed); np.random.seed(seed)
    N, dev = data["n_assets"], args.device
    obs_dim = 2 * N + 1
    agent = (PPO(obs_dim, N, head="gaussian", lr=args.ppo_lr, device=dev) if kind == "ppo"
             else DDPG(obs_dim, N, actor_lr=args.ddpg_lr, critic_lr=args.ddpg_lr,
                       batch=args.llc_batch, buffer=args.buffer, device=dev))
    e = TradingEnv(data["train"]["prices"], data["train"]["fr"], seed=seed)
    obs = e.state()
    gstep, best, best_state, hist = 0, -9e9, None, []
    t0 = time.time()

    while gstep < args.timesteps:
        if kind == "ppo":
            O, A, LP, R, D, V = [], [], [], [], [], []
            for _ in range(args.rollout):
                a, raw, logp, val = agent.act(obs)
                _, r, done, _ = e.step(a)
                O.append(obs); A.append(raw); LP.append(logp); R.append(r); D.append(float(done)); V.append(val)
                gstep += 1
                if done:
                    e.reset()
                obs = e.state()
            _, _, _, last_val = agent.act(obs)
            agent.update(O, A, LP, R, D, V, last_val)
        else:
            for _ in range(args.rollout):
                noise = args.noise * max(0.0, 1.0 - gstep / (0.6 * args.timesteps))
                a = (np.random.uniform(-1, 1, N) if gstep < args.warmup
                     else agent.act(obs, noise=noise))
                _, r, done, _ = e.step(a)
                obs2 = e.state()
                agent.buf.add(obs, a, r, obs2, float(done))
                gstep += 1
                if done:
                    e.reset()
                obs = e.state()
            if gstep >= args.warmup:
                agent.update(steps=max(1, args.rollout // args.update_every))

        if gstep % args.eval_every < args.rollout:
            m = evaluate(make_flat_stepper(agent, kind), data, "valid", seed)
            hist.append({"step": gstep, "valid_sharpe": m["sharpe"]})
            if m["sharpe"] > best:
                best = m["sharpe"]
                net = agent.pi if kind == "ppo" else agent.actor
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if args.verbose:
                print(f"  [{seed}:{kind}] step {gstep:6d} valid sharpe {m['sharpe']:+.3f} "
                      f"best {best:+.3f} ({time.time()-t0:.0f}s)", flush=True)
    if best_state:
        (agent.pi if kind == "ppo" else agent.actor).load_state_dict(best_state)
    return make_flat_stepper(agent, kind), hist, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["hrt", "ppo", "ddpg"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panel", default=os.path.join(ART, "panel.npz"))
    ap.add_argument("--signal", choices=["causal", "paper", "shuffle", "none"],
                    default="causal")
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=10_000)
    ap.add_argument("--eval_every", type=int, default=25_000)
    ap.add_argument("--noise", type=float, default=0.2)
    # one gradient step per `update_every` env steps, at `llc_batch` samples each:
    # 256 samples/env-step either way, but far fewer kernel launches
    ap.add_argument("--llc_batch", type=int, default=1024)
    ap.add_argument("--update_every", type=int, default=4)
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--ppo_lr", type=float, default=3e-4)
    ap.add_argument("--ddpg_lr", type=float, default=1e-3)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.001)
    ap.add_argument("--alpha_unit", choices=["step", "episode"], default="step")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log_trades", action="store_true")
    args = ap.parse_args()

    frfile = os.path.join(ART, "fr_paper.npz" if args.signal == "paper" else "fr_causal.npz")
    data = build_data(args.panel, frfile, args.signal)
    t0 = time.time()
    if args.agent == "hrt":
        stepper, hist, best = train_hrt(data, args, args.seed)
    else:
        stepper, hist, best = train_flat(data, args, args.seed, args.agent)

    res = {"agent": args.agent, "seed": args.seed, "signal": args.signal,
           "valid_sharpe": best, "train_seconds": time.time() - t0, "history": hist}
    for k in ("test2021", "test2022"):
        res[k] = evaluate(stepper, data, k, args.seed)
        tm = res[k].pop("_trade_matrix")
        if args.log_trades:
            np.save(os.path.join(ART, "runs",
                    f"trades_{args.agent}_{args.signal}{args.tag}_s{args.seed}_{k}.npy"), tm)
    name = f"{args.agent}_{args.signal}{args.tag}_s{args.seed}.json"
    os.makedirs(os.path.join(ART, "runs"), exist_ok=True)
    json.dump(res, open(os.path.join(ART, "runs", name), "w"))
    print(f"{args.agent} s{args.seed}: "
          f"2021 cum {res['test2021']['cum_return']:+.4f} sharpe {res['test2021']['sharpe']:+.4f} | "
          f"2022 cum {res['test2022']['cum_return']:+.4f} sharpe {res['test2022']['sharpe']:+.4f} | "
          f"{res['train_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
