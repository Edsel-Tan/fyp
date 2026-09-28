#!/usr/bin/env python3
"""Does the checkpoint rule select learning, or panel luck? (RESULTS.md sec.10 item 3)

sec.4.1 found validation Sharpe -- the statistic hrt/train.py keeps the best
checkpoint on -- *higher* on null panels than on the real one for every leak-free
arm. That says the rule is not measuring learning; it does not say what a better
rule would do. hrt/run_probe.sh reran the four leak-free arms on the real panel and
two null panels with `--probe`, which scores every eval point on the validation
window and on both test years and keeps its weights. So any selection rule can now
be applied after the fact to the same 20 checkpoints per run, and judged on test.

Rules, all using validation data only:
  raw     argmax validation Sharpe                       what train.py does
  excess  argmax Sharpe of (agent - passive floor) daily returns on validation.
          The floor is a constant within a run, so subtracting its *Sharpe* could
          not change the argmax; subtracting its *returns* removes the market's
          drift from the statistic, which is the term sec.4.1 blames.
  final   the last checkpoint                           no selection at all
  oracle  argmax test excess                            an upper bound, not a rule

Each is scored as test cumulative return in excess of the same panel's in-environment
passive floor (sec.4.1's metric). Two diagnostics sit beside it:
  * within-run rank correlation between the validation statistic and test excess
    across the 20 eval points -- if selection works, it is positive on the real
    panel and ~0 on the nulls, where there is nothing to learn;
  * the null-panel gain of each rule over `final`, which is pure selection noise
    and so is the yardstick any real-panel gain has to clear.

    python analysis/ckpt_rules.py
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, ttest_1samp

HERE = os.path.dirname(os.path.abspath(__file__))
HRT = os.path.join(HERE, os.pardir, "hrt")
ART = os.path.join(HRT, "artifacts")
sys.path.insert(0, HRT)
PANELS = {"real": ("panel.npz", "fr_causal.npz", "passive.json"),
          "null0": ("panel_synth.npz", "fr_synth_causal.npz", "passive_synth_s0.json"),
          "null1": ("panel_synth_s1.npz", "fr_synth_causal_s1.npz", "passive_synth_s1.json")}
TESTS = ("test2021", "test2022")


def rets(v):
    v = np.asarray(v, float)
    return v[1:] / v[:-1] - 1.0


def sharpe(x):
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(252)) if s > 0 else 0.0


def floors():
    """Passive floor value paths on validation and both test years, per panel."""
    cache = os.path.join(ART, "runs_probe", "floors.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    from passive import buy_and_hold
    from train import build_data
    out = {}
    for k, (pn, fr, _) in PANELS.items():
        d = build_data(os.path.join(ART, pn), os.path.join(ART, fr), "causal")
        out[k] = {per: buy_and_hold(d[per]["prices"], d[per]["fr"])["values"]
                  for per in ("valid",) + TESTS}
    json.dump(out, open(cache, "w"))
    return out


def main():
    F = floors()
    rows, corr = [], []
    for kind in PANELS:
        for f in sorted(glob.glob(os.path.join(ART, "runs_probe", kind, "*.json"))):
            stem = os.path.basename(f)[:-5]
            arm, seed = re.match(r"(.+)_s(\d+)$", stem).groups()
            r = json.load(open(f))
            H = [h for h in r["history"] if "probe" in h]
            if len(H) < 3:
                continue
            fv = rets(F[kind]["valid"])
            vs_raw = np.array([h["probe"]["valid"]["sharpe"] for h in H])
            vs_exc = np.array([sharpe(rets(h["probe"]["valid"]["values"]) - fv) for h in H])
            te = {per: np.array([h["probe"][per]["cum_return"] for h in H])
                       - (F[kind][per][-1] / F[kind][per][0] - 1) for per in TESTS}
            pick = {"raw": int(np.argmax(vs_raw)), "excess": int(np.argmax(vs_exc)),
                    "final": len(H) - 1}
            for per in TESTS:
                pick_o = int(np.argmax(te[per]))
                for rule, i in list(pick.items()) + [("oracle", pick_o)]:
                    rows.append(dict(kind=kind, arm=arm, seed=int(seed), period=per,
                                     rule=rule, step=H[i]["step"], excess=float(te[per][i])))
                corr.append(dict(kind=kind, arm=arm, seed=int(seed), period=per,
                                 rho_raw=spearmanr(vs_raw, te[per])[0],
                                 rho_excess=spearmanr(vs_exc, te[per])[0]))
            # consistency check: our 'raw' pick must match what train.py itself kept
            assert abs(vs_raw[pick["raw"]] - r["valid_sharpe"]) < 1e-9, stem
    R, C = pd.DataFrame(rows), pd.DataFrame(corr)
    if R.empty:
        print("no probe runs yet"); return
    R["group"] = np.where(R.kind == "real", "real", "null")
    C["group"] = np.where(C.kind == "real", "real", "null")
    pd.set_option("display.width", 200)

    print("runs:", R.groupby(["kind"]).apply(lambda g: g[["arm", "seed"]].drop_duplicates().shape[0],
                                             include_groups=False).to_dict())
    print("\n== test excess over own-panel floor, by selection rule (mean over seeds, t) ==")
    for per in TESTS:
        print(f"\n{per}")
        t = (R[R.period == per].groupby(["group", "arm", "rule"]).excess
             .agg(["mean", "count", lambda x: ttest_1samp(x, 0).statistic if len(x) > 2 else np.nan])
             .rename(columns={"<lambda_0>": "t"}).round(4))
        print(t.unstack("rule").to_string())

    print("\n== gain of each rule over 'final' (paired by run) ==")
    W = R.pivot_table(index=["group", "kind", "arm", "seed", "period"], columns="rule",
                      values="excess").reset_index()
    out = {}
    for g in ("real", "null"):
        for rule in ("raw", "excess", "oracle"):
            x = (W[W.group == g][rule] - W[W.group == g]["final"]).dropna()
            tt = ttest_1samp(x, 0).statistic if len(x) > 2 else np.nan
            out[f"{g}_{rule}_minus_final"] = dict(mean=float(x.mean()), t=float(tt), n=int(len(x)))
            print(f"  {g:>4} {rule:>6} - final: {x.mean():+.4f}  (t {tt:+.2f}, n {len(x)})")

    print("\n== within-run Spearman(validation statistic, test excess) across eval points ==")
    cc = C.groupby(["group", "arm"])[["rho_raw", "rho_excess"]].agg(["mean", "count"]).round(3)
    print(cc.to_string())
    for g in ("real", "null"):
        for col in ("rho_raw", "rho_excess"):
            x = C[C.group == g][col].dropna()
            out[f"{g}_{col}"] = dict(mean=float(x.mean()),
                                     t=float(ttest_1samp(x, 0).statistic) if len(x) > 2 else None,
                                     n=int(len(x)))
            print(f"  {g:>4} {col:>10}: mean {x.mean():+.3f}  (t {out[f'{g}_{col}']['t'] or float('nan'):+.2f}, n {len(x)})")

    res = dict(summary=out, rows=R.to_dict("records"), corr=C.to_dict("records"))
    p = os.path.join(ART, "ckpt_rules.json")
    json.dump(res, open(p, "w"), indent=1, default=float)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
