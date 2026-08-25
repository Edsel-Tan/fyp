#!/usr/bin/env python3
"""Can the high-level controller learn to follow its own signal at all?

Trains the PPO HLC on the directional-alignment reward alone (alpha held at 1),
with the LLC replaced by a fixed mid-size trade. If PPO is sound, the policy
should converge towards a_i = sign(fr_i) and mean alignment should climb --
especially under the leaky signal, where sign(fr) is nearly always correct.
A flat curve would indict the agent rather than the reward schedule.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import TradingEnv
from agents import PPO
from train import build_data

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", choices=["causal", "paper"], default="paper")
    ap.add_argument("--timesteps", type=int, default=60000)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    frfile = os.path.join(ART, "fr_paper.npz" if a.signal == "paper" else "fr_causal.npz")
    data = build_data(os.path.join(ART, "panel.npz"), frfile, a.signal)
    N = data["n_assets"]
    hlc = PPO(N, N, head="multi", device=a.device)
    e = TradingEnv(data["train"]["prices"], data["train"]["fr"], seed=0)
    obs = e.hlc_state()

    print(f"signal={a.signal}   N={N}")
    print(f"{'step':>8}{'mean align':>13}{'match sign(fr)':>16}{'frac hold':>11}{'entropy':>10}")
    g = 0
    while g < a.timesteps:
        O, A, LP, R, D, V, AD = [], [], [], [], [], [], []
        aligns, matches, holds = [], [], []
        for _ in range(a.rollout):
            ah, raw, logp, val = hlc.act(obs)
            _, _, done, info = e.step_hier(ah, np.zeros(N))   # fixed mid-size trade
            aligns.append(info["align"].mean())
            matches.append(float((ah == np.sign(np.rint(obs))).mean()))
            holds.append(float((ah == 0).mean()))
            O.append(obs); A.append(raw); LP.append(logp)
            R.append(float(info["align"].mean()))             # alpha = 1 throughout
            AD.append(info["align"].copy())                   # per-stock credit
            D.append(float(done)); V.append(val)
            g += 1
            if done:
                e.reset()
            obs = e.hlc_state()
        _, _, _, last_val = hlc.act(obs)
        hlc.update(O, A, LP, R, D, V, last_val, adv_dim=np.array(AD))
        with torch.no_grad():
            ent = hlc._dist(torch.as_tensor(np.array(O[:256]), dtype=torch.float32,
                                            device=a.device)).entropy().sum(-1).mean().item()
        print(f"{g:>8}{np.mean(aligns):>+13.4f}{np.mean(matches):>16.4f}"
              f"{np.mean(holds):>11.4f}{ent/N:>10.4f}")


if __name__ == "__main__":
    main()
