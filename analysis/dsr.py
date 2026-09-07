#!/usr/bin/env python3
"""Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) over the HRT sweep.

PAPERS.md [8] calls this non-negotiable for the write-up: every Sharpe quoted in
hrt/README.md is a maximum over a search of K trials, and the standard estimator
also ignores skew and kurtosis. This script applies the published correction.

  PSR(SR*) = Phi[ (SR - SR*) sqrt(n-1) / sqrt(1 - g3 SR + (g4-1)/4 SR^2) ]
  SR*_0    = sigma_SR [ (1-gamma) Phi^-1(1 - 1/K) + gamma Phi^-1(1 - 1/(K e)) ]
  DSR      = PSR(SR*_0)

SR here is the *per-observation* Sharpe of the daily return series, not the
repo's annualised-return / annualised-vol convention; both are reported so the
README numbers stay locatable.
"""
import glob, json, math, os, sys
import numpy as np
from scipy.stats import norm, skew, kurtosis

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, os.pardir, "hrt", "artifacts")
GAMMA = 0.5772156649015329          # Euler-Mascheroni


def rets(values):
    v = np.asarray(values, float)
    return v[1:] / v[:-1] - 1.0


def psr(sr, sr_star, n, g3, g4):
    """Probability the true Sharpe exceeds sr_star, given observed sr over n obs."""
    den = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if den <= 0:
        return float("nan")
    return float(norm.cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(den)))


def expected_max_sr(sr_var, K):
    """E[max SR] over K independent trials with variance sr_var, under a zero-SR null."""
    if K < 2:
        return 0.0
    a = norm.ppf(1.0 - 1.0 / K)
    b = norm.ppf(1.0 - 1.0 / (K * math.e))
    return math.sqrt(sr_var) * ((1.0 - GAMMA) * a + GAMMA * b)


def load(dirs):
    out = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(ART, d, "*.json"))):
            r = json.load(open(f))
            out.append((os.path.basename(d), os.path.basename(f)[:-5], r))
    return out


def main():
    # K must count every run ever aggregated -- runs/ AND the discarded runs_invalid/,
    # since the surviving configuration was chosen after seeing both.
    runs = load(["runs", "runs_invalid"])
    valid = [r for r in runs if r[0] == "runs"]
    K = len(runs)
    print(f"trials K = {K}  ({len(valid)} reported in runs/, {K - len(valid)} in runs_invalid/)")

    for period in ("test2021", "test2022"):
        print(f"\n=== {period} ===")
        # arm-level SR variance across all K trials, per Bailey-Lopez de Prado
        srs = []
        for _, _, r in runs:
            if period in r:
                x = rets(r[period]["values"])
                srs.append(x.mean() / x.std(ddof=1))
        sr_var = float(np.var(srs, ddof=1))
        sr_star = expected_max_sr(sr_var, K)
        print(f"sigma_SR across trials = {math.sqrt(sr_var):.4f} (daily)   "
              f"SR*_0 = {sr_star:.4f} daily = {sr_star*math.sqrt(252):.3f} annualised")

        # aggregate by arm (mean over seeds), then deflate the per-seed maximum
        arms = {}
        for src, name, r in runs:
            if period not in r:
                continue
            arm = name.rsplit("_s", 1)[0] + ("" if src == "runs" else " [invalid]")
            arms.setdefault(arm, []).append(r[period])

        print(f"\n{'arm':<28}{'n':>3}{'SR_ann':>9}{'SR_day':>9}{'skew':>8}{'kurt':>8}"
              f"{'PSR(0)':>9}{'DSR':>9}")
        rows = []
        for arm, rs in sorted(arms.items()):
            best = max(rs, key=lambda m: m["sharpe"])
            x = rets(best["values"])
            n = len(x)
            sr = x.mean() / x.std(ddof=1)
            g3, g4 = float(skew(x, bias=False)), float(kurtosis(x, fisher=False, bias=False))
            rows.append((arm, len(rs), best["sharpe"], sr, g3, g4,
                         psr(sr, 0.0, n, g3, g4), psr(sr, sr_star, n, g3, g4)))
        for a, m, sa, sr, g3, g4, p0, d in sorted(rows, key=lambda r: -r[2]):
            print(f"{a:<28}{m:>3}{sa:>9.2f}{sr:>9.4f}{g3:>8.2f}{g4:>8.2f}{p0:>9.3f}{d:>9.3f}")
        json.dump([dict(zip(("arm", "n_seeds", "sharpe_ann_repo", "sharpe_daily",
                             "skew", "kurtosis", "psr_vs_0", "dsr"), r)) for r in rows],
                  open(os.path.join(ART, f"dsr_{period}.json"), "w"), indent=1)

    print(f"\nwrote {ART}/dsr_test20*.json")


if __name__ == "__main__":
    main()
