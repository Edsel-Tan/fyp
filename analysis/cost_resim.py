#!/usr/bin/env python3
"""Exact re-simulation of the agents under other cost rates (RESULTS.md sec.10 item 4).

analysis/cost_rerank.py had to recharge each run's recorded trade schedule to first
order, because the policies were thrown away:  r(rho) = r(10bp) + (10bp - rho) *
mean turnover. hrt/run_probe.sh keeps the weights, so the same question can now be
answered exactly: rebuild the checkpoint train.py itself selects (best validation
Sharpe), and replay both test years through TradingEnv at each flat rate. The policy
is held fixed -- it was trained at 10 bp -- but the cash path, the fills that cash
permits, and the compounding all respond to the new rate, which is precisely what
the recharge leaves out.

The comparison that matters is MACE's claim, the ranking: for each rate, rank arms
by seed-mean Sharpe under the exact replay and under the recharge, and count the
arms that move against the 10 bp ranking in each.

    python analysis/cost_resim.py                 # the real panel
    python analysis/cost_resim.py --kind null0
"""
import argparse, glob, json, os, re, sys
import numpy as np, torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
HRT = os.path.join(HERE, os.pardir, "hrt")
ART = os.path.join(HRT, "artifacts")
sys.path.insert(0, HRT); sys.path.insert(0, HERE)
from env import TradingEnv, metrics
from agents import PPO, DDPG
from train import build_data, make_hrt_stepper, make_flat_stepper
from cost_rerank import RATES, recharge, sharpe as sharpe_r
from ckpt_rules import PANELS


def rebuild(stem, N, sd):
    """The checkpoint's policy nets, on CPU, with no replay buffer."""
    agent = stem.split("_")[0]
    if agent == "hrt":
        hlc = PPO(N, N, head="multi", device="cpu")
        llc = DDPG(3 * N + 1, N, buffer=1, device="cpu")
        hlc.pi.load_state_dict(sd["hlc_pi"]); llc.actor.load_state_dict(sd["llc_actor"])
        return make_hrt_stepper(hlc, llc)
    obs = 2 * N + 1
    if agent == "ppo":
        a = PPO(obs, N, head="gaussian", device="cpu"); a.pi.load_state_dict(sd["pi"])
    else:
        a = DDPG(obs, N, buffer=1, device="cpu"); a.actor.load_state_dict(sd["actor"])
    return make_flat_stepper(a, agent)


def replay(step, data, key, seed, cost):
    e = TradingEnv(data[key]["prices"], data[key]["fr"], seed=seed, cost=cost)
    done = False
    while not done:
        done = step(e)
    return np.asarray(e.values), float(np.mean(e.turnover))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="real", choices=list(PANELS))
    a = ap.parse_args()
    torch.set_num_threads(2)
    pn, fr, _ = PANELS[a.kind]
    data = build_data(os.path.join(ART, pn), os.path.join(ART, fr), "causal")
    N = data["n_assets"]
    rows = []
    for f in sorted(glob.glob(os.path.join(ART, "runs_probe", a.kind, "*.json"))):
        stem = os.path.basename(f)[:-5]
        arm, seed = re.match(r"(.+)_s(\d+)$", stem).groups(); seed = int(seed)
        r = json.load(open(f))
        H = [h for h in r["history"] if "probe" in h]
        best = max(H, key=lambda h: h["valid_sharpe"])
        ck = os.path.join(ART, "runs_probe", a.kind, "ckpt", stem, f"step{best['step']:07d}.pt")
        step = rebuild(stem, N, torch.load(ck, map_location="cpu"))
        for per in ("test2021", "test2022"):
            base_v, base_to = replay(step, data, per, seed, 0.001)
            # the replay at the training rate should reproduce the run's own test number.
            # Policies were trained on GPU and replay on CPU; float differences can flip
            # an argmax or a share rounding, so a few runs drift. Record it, do not hide it:
            # both the exact and the recharged paths start from this same CPU replay.
            gap = base_v[-1] / base_v[0] - 1 - r[per]["cum_return"]
            for name, rho in RATES.items():
                v, _ = replay(step, data, per, seed, rho)
                vr = recharge(base_v, base_to, rho)
                rows.append(dict(arm=arm, seed=seed, period=per, rate=name,
                                 exact=metrics(v)["sharpe"], recharge=metrics(vr)["sharpe"],
                                 exact_daily=sharpe_r(v), recharge_daily=sharpe_r(vr),
                                 turnover=base_to, replay_gap=gap))
        print(f"  {stem} step {best['step']}", flush=True)

    import pandas as pd
    D = pd.DataFrame(rows)
    out = {"kind": a.kind, "n_runs": int(D[["arm", "seed"]].drop_duplicates().shape[0])}
    g = D.drop_duplicates(["arm", "seed", "period"]).replay_gap.abs()
    out["replay_exact"] = int((g < 1e-9).sum()); out["replay_n"] = int(len(g))
    out["replay_max_gap"] = float(g.max())
    print(f"replay reproduces {out['replay_exact']}/{out['replay_n']} recorded test returns to 1e-9; "
          f"max gap {out['replay_max_gap']:.2e} (GPU-trained, CPU-replayed)")
    for per in ("test2021", "test2022"):
        M = D[D.period == per].groupby(["rate", "arm"])[["exact", "recharge"]].mean()
        print(f"\n{per}: seed-mean Sharpe (paper convention)\n{M.unstack('rate').round(3).to_string()}")
        rk = {m: {rate: M.loc[rate][m].rank(ascending=False).to_dict() for rate in RATES}
              for m in ("exact", "recharge")}
        moved = {m: {rate: int(sum(rk[m][rate][x] != rk[m]["paper 10bp"][x] for x in rk[m][rate]))
                     for rate in RATES} for m in rk}
        agree = {rate: float(spearmanr(M.loc[rate].exact, M.loc[rate].recharge)[0]) for rate in RATES}
        gap = {rate: float((M.loc[rate].exact - M.loc[rate].recharge).abs().max()) for rate in RATES}
        print(f"  arms moving rank vs 10bp  exact {moved['exact']}  recharge {moved['recharge']}")
        print(f"  Spearman(exact, recharge) {agree}\n  max |exact - recharge| Sharpe {gap}")
        out[per] = dict(moved=moved, agree=agree, max_gap=gap, ranks=rk,
                        table=M.reset_index().to_dict("records"))
    p = os.path.join(ART, f"cost_resim_{a.kind}.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
