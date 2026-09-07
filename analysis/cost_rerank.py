#!/usr/bin/env python3
"""Does the cost model reorder the HRT agents? (MACE, PAPERS.md [2], on equities.)

The trained policies were not checkpointed and each run cost ~2 h, so the agents
cannot be re-simulated under a different cost model. What the run records do carry
is the realised net value path, the mean per-step turnover and the total cost paid
at the 10 bp rate the environment charged. That is enough to recharge the same
trade schedule at a different flat rate to first order:

    r_net(rho)_t = r_net(10bp)_t + (0.001 - rho) * turnover_t

with turnover held at its run mean, since only the mean is recorded. The
approximation ignores the day-to-day variation of turnover and the second-order
effect of a different cost path on the value the agent compounds; it is used only
to ask whether the *ranking* is stable, which is MACE's claim, not to restate
levels. Gross (zero-cost) is the rho = 0 case and is exact in mean.
"""
import glob, json, os
import numpy as np

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "hrt", "artifacts")
RATES = {"gross (0bp)": 0.0, "paper 10bp": 0.001, "30bp": 0.003, "60bp": 0.006}


def recharge(values, turnover, rho, base=0.001):
    v = np.asarray(values, float)
    r = v[1:] / v[:-1] - 1.0 + (base - rho) * turnover
    out = np.empty_like(v)
    out[0] = v[0]
    out[1:] = v[0] * np.cumprod(1 + r)
    return out


def sharpe(v, periods=252):
    r = v[1:] / v[:-1] - 1.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def main():
    runs = {}
    for f in sorted(glob.glob(os.path.join(ART, "runs", "*.json"))):
        d = json.load(open(f))
        arm = os.path.basename(f)[:-5].rsplit("_s", 1)[0]
        runs.setdefault(arm, []).append(d)

    for period in ("test2021", "test2022"):
        print(f"\n=== {period}: Sharpe under each flat cost rate (mean over seeds) ===")
        print(f"{'arm':<20}{'turnover':>10}" + "".join(f"{k:>14}" for k in RATES))
        tbl = {}
        for arm, ds in sorted(runs.items()):
            row, turns = {}, []
            for rate_name, rho in RATES.items():
                s = []
                for d in ds:
                    m = d[period]
                    s.append(sharpe(recharge(m["values"], m["turnover"], rho)))
                    turns.append(m["turnover"])
                row[rate_name] = float(np.mean(s))
            tbl[arm] = row
            print(f"{arm:<20}{np.mean(turns):>10.3f}" +
                  "".join(f"{row[k]:>+14.2f}" for k in RATES))

        arms = list(tbl)
        ranks = {k: {a: i + 1 for i, a in
                     enumerate(sorted(arms, key=lambda a: -tbl[a][k]))} for k in RATES}
        print(f"\n{'arm':<20}" + "".join(f"{k:>14}" for k in RATES))
        for a in arms:
            print(f"{a:<20}" + "".join(f"{ranks[k][a]:>14d}" for k in RATES))
        base = "paper 10bp"
        for k in RATES:
            if k == base:
                continue
            moved = sum(ranks[k][a] != ranks[base][a] for a in arms)
            print(f"  '{k}' vs '{base}': {moved}/{len(arms)} arms move rank")


if __name__ == "__main__":
    main()
