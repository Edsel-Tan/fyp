#!/usr/bin/env python3
"""Deflated Sharpe Ratio over the crypto strategy sweep ([8]).

RESULTS.md sec.2.2 applies Bailey & Lopez de Prado's correction to the equity RL
sweep and finds no leak-free arm survives it. Section 5.3 then quotes crypto
Sharpes -- `forecast-causal top5` at +2.07 gross -- with no deflation at all,
which is the exact error sec.6 accuses the reading list of. This closes that gap
on the repo's own results.

The trial count is the argument, so it is reported at two readings:

  K = n strategies          a researcher who fixed the cost model and searched
                            over strategies, reporting the best;
  K = n strategies x models a researcher who also chose the cost model after
                            seeing the results -- which is what sec.2.3 does when
                            it reports the ranking under four models.

sigma_SR is the dispersion of per-bar Sharpe across the trials in the search,
matching analysis/dsr.py.

    python crypto/dsr.py                    # needs crypto/artifacts/strategy_paths.npz
"""
import argparse, json, math, os, sys
import numpy as np
from scipy.stats import skew, kurtosis

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from analysis.dsr import psr, expected_max_sr, rets              # noqa: E402

ART = os.path.join(HERE, "artifacts")
BARS_PER_YEAR = 24 * 365


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default=os.path.join(ART, "strategy_paths.npz"))
    a = ap.parse_args()
    if not os.path.exists(a.paths):
        sys.exit(f"{a.paths} missing -- run crypto/strategy.py first")
    z = np.load(a.paths, allow_pickle=True)
    keys = [k for k in z.files if k != "dates"]
    models = sorted({k.split("|", 1)[0] for k in keys})
    strats = sorted({k.split("|", 1)[1] for k in keys})
    dates = z["dates"].astype(str)
    print(f"window {dates[0]} .. {dates[-1]}   {len(dates)} bars   "
          f"{len(strats)} strategies x {len(models)} cost models = {len(keys)} trials")

    per = {k: rets(z[k].astype(np.float64)) for k in keys}
    out = {}
    for cname in models:
        sub = {s: per[f"{cname}|{s}"] for s in strats if f"{cname}|{s}" in per}
        srs = np.array([r.mean() / r.std(ddof=1) for r in sub.values()])
        sr_var = float(np.var(srs, ddof=1))
        # Under a cost model a few strategies lose almost the whole book, so the
        # sample sd of SR across trials is set by two outliers and SR*_0 becomes
        # uninformative. A robust scale is reported alongside; the gross panel,
        # where the trials are comparable, is the one that discriminates.
        q1, q3 = np.percentile(srs, [25, 75])
        sr_var_rob = float(((q3 - q1) / 1.349) ** 2)
        K1, K2 = len(sub), len(keys)
        s1, s2 = expected_max_sr(sr_var, K1), expected_max_sr(sr_var, K2)
        s1r = expected_max_sr(sr_var_rob, K1)
        print(f"\n=== cost model: {cname} ===")
        print(f"sigma_SR across the {K1} strategies = {math.sqrt(sr_var):.5f} per bar "
              f"(robust IQR/1.349 = {math.sqrt(sr_var_rob):.5f})")
        print(f"SR*_0 = {s1*math.sqrt(BARS_PER_YEAR):.3f} annualised (K={K1}), "
              f"{s2*math.sqrt(BARS_PER_YEAR):.3f} (K={K2}), "
              f"{s1r*math.sqrt(BARS_PER_YEAR):.3f} (K={K1}, robust sigma)")
        print(f"{'strategy':<40}{'SR_ann':>9}{'skew':>8}{'kurt':>8}"
              f"{'PSR(0)':>9}{f'DSR K={K1}':>11}{f'DSR K={K2}':>11}{'DSR robust':>12}")
        rows = []
        for s, r in sub.items():
            n = len(r)
            sr = r.mean() / r.std(ddof=1)
            g3 = float(skew(r, bias=False))
            g4 = float(kurtosis(r, fisher=False, bias=False))
            rows.append((s, sr * math.sqrt(BARS_PER_YEAR), g3, g4,
                         psr(sr, 0.0, n, g3, g4),
                         psr(sr, s1, n, g3, g4), psr(sr, s2, n, g3, g4),
                         psr(sr, s1r, n, g3, g4)))
        for s, sa, g3, g4, p0, d1, d2, dr in sorted(rows, key=lambda r: -r[1]):
            print(f"{s:<40}{sa:>+9.2f}{g3:>8.2f}{g4:>8.2f}{p0:>9.3f}"
                  f"{d1:>11.3f}{d2:>11.3f}{dr:>12.3f}")
        surv1 = [r[0] for r in rows if r[5] > 0.95]
        surv2 = [r[0] for r in rows if r[6] > 0.95]
        print(f"survive DSR>0.95 at K={K1}: {len(surv1)}/{K1} {surv1}")
        print(f"survive DSR>0.95 at K={K2}: {len(surv2)}/{K1} {surv2}")
        out[cname] = dict(sigma_sr=math.sqrt(sr_var),
                          sr_star_K1=s1 * math.sqrt(BARS_PER_YEAR),
                          sr_star_K2=s2 * math.sqrt(BARS_PER_YEAR),
                          sr_star_K1_robust=s1r * math.sqrt(BARS_PER_YEAR),
                          rows=[dict(zip(("strategy", "sharpe_ann", "skew", "kurtosis",
                                          "psr_vs_0", "dsr_K1", "dsr_K2", "dsr_robust"),
                                         r)) for r in rows])
    f = os.path.join(ART, "dsr_strategy.json")
    json.dump(out, open(f, "w"), indent=1)
    print("\nwrote", f)


if __name__ == "__main__":
    main()
