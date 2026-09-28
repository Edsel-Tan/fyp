#!/usr/bin/env python3
"""Falsification audit of the RL sweep (PAPERS.md [10]), at the agent level.

analysis/falsify.py audits the *forecaster* on synthetic panels whose returns are
a martingale difference. This audits the *agents*: hrt/slurm/05_sweep_null.sbatch
retrains the whole 25-run sweep on those same panels, at the identical step
budget, and hrt/passive.py measures a do-nothing floor inside the same
environment on each one.

Every comparison here is an **excess over the passive floor of the panel the run
was trained on**. That differences out the panel's own realised drift, which is
large and varies across null panels (2022 floor: -0.16 to +0.55), and it is the
only benchmark an RL agent can be held to -- the agent deploys capital, so the
zero-return null the DSR assumes is the wrong centre.

Four questions, in order:
  1. Do null-panel agents beat their own floor? Split by label timing.
  2. Is validation Sharpe -- the statistic train.py checkpoints on -- larger on
     the null than on the real panel?
  3. Does the null sweep reproduce the *ordering* of arms that RESULTS.md §2
     reports on the real panel?
  4. Used as a direct Monte Carlo of max-Sharpe-over-K-trials, does the null
     sweep agree with the Deflated Sharpe Ratio's analytic SR*_0 (analysis/dsr.py)?

  python analysis/null_sweep.py
"""
import argparse, glob, json, math, os, re
import numpy as np
from scipy.stats import norm, spearmanr, ttest_1samp, ttest_ind

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, os.pardir, "hrt", "artifacts")
PERIODS = ("test2021", "test2022")
GAMMA = 0.5772156649015329
ANN = math.sqrt(252)


def leaky(arm):
    """Arms trained on the paper's label timing, which leaks the same-bar close."""
    return "paper" in arm


def load_runs(d):
    out = []
    for f in sorted(glob.glob(os.path.join(ART, d, "*.json"))):
        stem = os.path.basename(f)[:-5]
        if "_smoke" in stem:
            continue
        m = re.search(r"_s(\d+)$", stem)
        if not m:
            continue
        out.append((stem[:m.start()], int(m.group(1)), json.load(open(f))))
    return out


def daily_sr(values):
    v = np.asarray(values, float)
    x = v[1:] / v[:-1] - 1.0
    return float(x.mean() / x.std(ddof=1))


def expected_max_sr(sr_var, K):
    """Bailey-Lopez de Prado E[max SR] over K trials under a zero-SR null."""
    if K < 2:
        return 0.0
    a, b = norm.ppf(1.0 - 1.0 / K), norm.ppf(1.0 - 1.0 / (K * math.e))
    return math.sqrt(sr_var) * ((1.0 - GAMMA) * a + GAMMA * b)


def tstat(x):
    return float(ttest_1samp(x, 0.0).statistic) if len(x) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npanels", type=int, default=4,
                    help="null panels the sweep cycled over; seed k trained on panel k %% npanels")
    ap.add_argument("--out", default=os.path.join(ART, "null_sweep.json"))
    a = ap.parse_args()

    floors = {int(re.search(r"_s(\d+)\.json$", f).group(1)): json.load(open(f))
              for f in sorted(glob.glob(os.path.join(ART, "passive_synth_s*.json")))}
    if not floors:
        raise SystemExit("no passive_synth_s*.json -- run 05_sweep_null.sbatch STAGE=prep")
    real_floor = json.load(open(os.path.join(ART, "passive.json")))
    null, real, invalid = load_runs("runs_null"), load_runs("runs"), load_runs("runs_invalid")
    print(f"null runs {len(null)} on {len(floors)} panels   real runs {len(real)}"
          f"   discarded {len(invalid)}")
    # a null run trained for fewer steps than its real counterpart would explain every
    # difference below, so this is asserted rather than eyeballed
    steps = {r["history"][-1]["step"] for _, _, r in null + real if r.get("history")}
    if len(steps) != 1:
        raise SystemExit(f"step budgets differ across runs: {sorted(steps)} -- "
                         "the null and real sweeps are not comparable")
    print(f"step budget, identical on both sides: {steps.pop():,}")

    def floor_of(kind, seed, period):
        return (floors[seed % a.npanels] if kind == "null" else real_floor)[period]

    # rows: one per (kind, arm, seed, period), excess over the matching floor
    rows = []
    for kind, runs in (("null", null), ("real", real)):
        for arm, seed, r in runs:
            for period in PERIODS:
                m, f = r[period], floor_of(kind, seed, period)
                rows.append(dict(kind=kind, arm=arm, seed=seed, period=period,
                                 cum=m["cum_return"], sharpe=m["sharpe"],
                                 turnover=m["turnover"], valid_sharpe=r["valid_sharpe"],
                                 exc_cum=m["cum_return"] - f["cum_return"],
                                 exc_sr=m["sharpe"] - f["sharpe"]))

    def pick(period, kind, pred=lambda arm: True, key="exc_cum"):
        return np.array([r[key] for r in rows
                         if r["period"] == period and r["kind"] == kind and pred(r["arm"])])

    out = {"n_null": len(null), "n_real": len(real), "n_invalid": len(invalid),
           "step_budget": sorted(steps), "periods": {}}

    # ---------------------------------------------------------------- Q1
    for period in PERIODS:
        fl = [floors[k][period]["cum_return"] for k in sorted(floors)]
        print("\n" + "=" * 118)
        print(f"{period}  --  cumulative return in excess of the passive floor of the SAME panel")
        print(f"null-panel floors: {', '.join(f'{x:+.3f}' for x in fl)}"
              f"   real floor: {real_floor[period]['cum_return']:+.3f}")
        print("=" * 118)
        print(f"{'arm':<16}{'n':>4}{'null excess':>20}{'t':>7}{'beat':>7}  |"
              f"{'n':>4}{'real excess':>20}{'t':>7}{'beat':>7}{'null=real p':>13}")
        print("-" * 118)
        per_arm = {}
        for arm in sorted({r["arm"] for r in rows}):
            n = pick(period, "null", lambda x, arm=arm: x == arm)
            r_ = pick(period, "real", lambda x, arm=arm: x == arm)
            p = float(ttest_ind(n, r_, equal_var=False).pvalue)
            print(f"{arm:<16}{len(n):>4}{f'{n.mean():+.4f}±{n.std(ddof=1):.3f}':>20}"
                  f"{tstat(n):>+7.2f}{f'{int((n > 0).sum())}/{len(n)}':>7}  |"
                  f"{len(r_):>4}{f'{r_.mean():+.4f}±{r_.std(ddof=1):.3f}':>20}"
                  f"{tstat(r_):>+7.2f}{f'{int((r_ > 0).sum())}/{len(r_)}':>7}{p:>13.3f}")
            per_arm[arm] = dict(null_exc=float(n.mean()), null_t=tstat(n),
                                null_beat=int((n > 0).sum()), null_n=len(n),
                                real_exc=float(r_.mean()), real_t=tstat(r_),
                                real_beat=int((r_ > 0).sum()), real_n=len(r_), p=p)
        print("-" * 118)
        groups = {}
        for lab, pred in (("leaky (paper timing)", leaky),
                          ("leak-free (causal timing)", lambda x: not leaky(x))):
            n, r_ = pick(period, "null", pred), pick(period, "real", pred)
            # the share of the real effect a panel with nothing to learn reproduces --
            # only meaningful when both sides point the same way
            share = (f"   null/real {n.mean() / r_.mean():.2f}"
                     if n.mean() > 0 and r_.mean() > 0 else "")
            print(f"{lab:<28} null {n.mean():+.4f} (t {tstat(n):+.2f}, {int((n > 0).sum())}/{len(n)})"
                  f"   real {r_.mean():+.4f} (t {tstat(r_):+.2f}, {int((r_ > 0).sum())}/{len(r_)}){share}")
            groups[lab] = dict(null=float(n.mean()), null_t=tstat(n),
                               null_beat=int((n > 0).sum()), null_n=len(n),
                               real=float(r_.mean()), real_t=tstat(r_),
                               real_beat=int((r_ > 0).sum()), real_n=len(r_))
        out["periods"][period] = dict(arms=per_arm, groups=groups,
                                      null_floors=fl,
                                      real_floor=real_floor[period]["cum_return"])

    # ---------------------------------------------------------------- Q2
    print("\n" + "=" * 118)
    print("validation Sharpe -- the statistic train.py checkpoints on (one value per run)")
    print("=" * 118)
    vs = {(r["kind"], r["arm"]): [] for r in rows}
    for r in rows:
        if r["period"] == PERIODS[0]:
            vs[(r["kind"], r["arm"])].append(r["valid_sharpe"])
    print(f"{'arm':<16}{'null mean':>12}{'null range':>20}{'real mean':>12}{'real range':>20}")
    print("-" * 118)
    for arm in sorted({r["arm"] for r in rows}):
        n, r_ = np.array(vs[("null", arm)]), np.array(vs[("real", arm)])
        print(f"{arm:<16}{n.mean():>+12.3f}{f'[{n.min():+.3f}, {n.max():+.3f}]':>20}"
              f"{r_.mean():>+12.3f}{f'[{r_.min():+.3f}, {r_.max():+.3f}]':>20}")
    out["valid_sharpe"] = {}
    for lab, pred in (("leaky", leaky), ("leak-free", lambda x: not leaky(x))):
        n = np.concatenate([v for (k, arm), v in vs.items() if k == "null" and pred(arm)])
        r_ = np.concatenate([v for (k, arm), v in vs.items() if k == "real" and pred(arm)])
        med = float(np.median(n))
        below = int((r_ < med).sum())
        pct = float(np.mean([100.0 * np.searchsorted(np.sort(n), x) / len(n) for x in r_]))
        t = float(ttest_ind(r_, n, equal_var=False).statistic)
        print(f"\n  {lab}: null {n.mean():+.3f} [{n.min():+.3f}, {n.max():+.3f}] (n={len(n)})"
              f"   real {r_.mean():+.3f} [{r_.min():+.3f}, {r_.max():+.3f}] (n={len(r_)})"
              f"   real-vs-null t {t:+.2f}")
        print(f"     {below}/{len(r_)} real runs below the null median ({med:+.3f});"
              f" mean percentile of a real run in the null distribution {pct:.1f}"
              f"  (50 = indistinguishable)")
        out["valid_sharpe"][lab] = dict(null_mean=float(n.mean()), real_mean=float(r_.mean()),
                                        null_median=med, real_below_null_median=below,
                                        real_n=len(r_), mean_percentile=pct, t=t)

    # ---------------------------------------------------------------- Q3
    print("\n" + "=" * 118)
    print("arm ordering -- does a panel with nothing to learn reproduce §2's ranking?")
    print("=" * 118)
    for period in PERIODS:
        arms = sorted({r["arm"] for r in rows})
        n = np.array([np.mean([r["exc_cum"] for r in rows if r["period"] == period
                               and r["kind"] == "null" and r["arm"] == arm]) for arm in arms])
        r_ = np.array([np.mean([r["exc_cum"] for r in rows if r["period"] == period
                                and r["kind"] == "real" and r["arm"] == arm]) for arm in arms])
        rho = spearmanr(n, r_)
        order_n = [arms[i] for i in np.argsort(-n)]
        order_r = [arms[i] for i in np.argsort(-r_)]
        print(f"\n{period}  Spearman rho = {rho.statistic:+.3f} (p = {rho.pvalue:.3f})")
        print(f"  null ranking: {' > '.join(order_n)}")
        print(f"  real ranking: {' > '.join(order_r)}")
        out["periods"][period]["rank_spearman"] = float(rho.statistic)
        out["periods"][period]["rank_p"] = float(rho.pvalue)
        out["periods"][period]["null_order"] = order_n
        out["periods"][period]["real_order"] = order_r

    # ---------------------------------------------------------------- Q4
    print("\n" + "=" * 118)
    print("the null sweep as a direct Monte Carlo of the Deflated Sharpe Ratio's SR*_0")
    print("=" * 118)
    for period in PERIODS:
        print(f"\n{period}")
        for lab, runs, pred in (("null, leak-free arms", null, lambda x: not leaky(x)),
                                ("null, leaky arms", null, leaky),
                                ("null, all arms", null, lambda x: True)):
            sr = np.array([daily_sr(r[period]["values"]) for arm, _, r in runs if pred(arm)])
            K = len(sr)
            ana = expected_max_sr(float(sr.var(ddof=1)), K)
            print(f"  {lab:<24} K={K:<3} sigma_SR {sr.std(ddof=1):.4f}"
                  f"   observed max {sr.max() * ANN:+.3f} ann"
                  f"   analytic E[max] {ana * ANN:+.3f} ann   ratio {sr.max() / ana:.2f}x"
                  f"   mean SR {sr.mean() * ANN:+.3f} ann")
            out["periods"][period][f"mc_{lab.split(', ')[1].replace(' ', '_')}"] = dict(
                K=K, sigma_sr=float(sr.std(ddof=1)), obs_max_ann=float(sr.max() * ANN),
                analytic_max_ann=float(ana * ANN), mean_ann=float(sr.mean() * ANN))
        # the DSR the real sweep actually faces, and who clears it
        all_real = np.array([daily_sr(r[period]["values"]) for _, _, r in real + invalid])
        K = len(all_real)
        srstar = expected_max_sr(float(all_real.var(ddof=1)), K)
        lf_real = np.array([daily_sr(r[period]["values"]) for arm, _, r in real if not leaky(arm)])
        lf_null = np.array([daily_sr(r[period]["values"]) for arm, _, r in null if not leaky(arm)])
        print(f"  real sweep K={K}   SR*_0 {srstar * ANN:+.3f} ann")
        print(f"    best leak-free REAL run {lf_real.max() * ANN:+.3f} ann"
              f"   best leak-free NULL run {lf_null.max() * ANN:+.3f} ann"
              f"   -> {'NULL WINS' if lf_null.max() > lf_real.max() else 'real wins'}")
        print(f"    leak-free null runs at or above the best leak-free real run: "
              f"{int((lf_null >= lf_real.max()).sum())}/{len(lf_null)}")
        out["periods"][period]["dsr"] = dict(
            K=K, sr_star_ann=float(srstar * ANN),
            best_leakfree_real_ann=float(lf_real.max() * ANN),
            best_leakfree_null_ann=float(lf_null.max() * ANN),
            null_ge_best_real=int((lf_null >= lf_real.max()).sum()), n_null_leakfree=len(lf_null))

        # non-parametric: the null sweep IS the reference distribution
        print(f"  empirical p-values (excess Sharpe over own-panel floor, vs the "
              f"{len(lf_null)} leak-free null runs)")
        ref = np.array([r["exc_sr"] for r in rows if r["period"] == period
                        and r["kind"] == "null" and not leaky(r["arm"])])
        pv = {}
        for arm in sorted({r["arm"] for r in rows}):
            e = np.array([r["exc_sr"] for r in rows if r["period"] == period
                          and r["kind"] == "real" and r["arm"] == arm])
            p_best, p_mean = float((ref >= e.max()).mean()), float((ref >= e.mean()).mean())
            print(f"    {arm:<16} best seed excess {e.max():+7.3f}  p {p_best:.3f}"
                  f"    seed mean excess {e.mean():+7.3f}  p {p_mean:.3f}")
            pv[arm] = dict(best_excess=float(e.max()), p_best=p_best,
                           mean_excess=float(e.mean()), p_mean=p_mean)
        out["periods"][period]["empirical_p"] = pv

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
