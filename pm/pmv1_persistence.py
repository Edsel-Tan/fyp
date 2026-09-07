#!/usr/bin/env python3
"""Does prediction-market "skill" persist out of sample? ([13], [15], [16])

RESULTS.md sec.3.1 shows the published sign-randomisation test over-rejects, and
that once the randomisation unit is corrected the flagged fraction falls to
chance. That is a statement about the *estimator*. It leaves the economic
question open, and the vendor tape could not answer it: fifteen weeks is not
enough calendar to form a view in one period and test it in another.

Polymarket-v1 is 3.5 years, so the decisive test is available. Split every
account's history at a cut date, rank accounts on the first half, and ask what
the second half does. A genuinely informed minority must persist; a tail
manufactured by a mis-calibrated null must not. This is out-of-sample by
construction and needs no null at all -- it is the same logic as a walk-forward
split, applied to traders instead of models.

Reported for each in-sample ranking:
  * mean out-of-sample z and PnL by in-sample z decile;
  * Spearman rank correlation between the two halves' z;
  * hit rate: of accounts flagged skilled in-sample, what fraction is flagged
    out-of-sample, against the base rate among all accounts;
  * the economic version: total out-of-sample PnL of the in-sample top decile,
    which is what copying them would have earned before costs.

z is computed at the *event* level throughout, since sec.3.1 shows that is the
only unit whose null is calibrated. `event` is neg_risk_market_id where the
market has one (ground truth, from the negative-risk contract) and condition_id
otherwise.

    python pm/pmv1_persistence.py --cut 2025-01-01
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)

# A market whose trading straddles the cut would put the *same* resolution outcome
# on both sides of the split: an account with a position in it appears skilled in
# both halves for one reason, which is leakage, not persistence. So events are
# assigned whole -- an event counts only if all of its trades fall on one side --
# and the two halves then share no outcome at all.
HALVES_SQL = """
WITH ev AS (
  SELECT event_id, min(ts) AS mn, max(ts) AS mx
  FROM read_parquet('{trades}') GROUP BY 1
), side AS (
  SELECT event_id, mx < {cut} AS in_sample FROM ev
  WHERE mx < {cut} OR mn >= {cut}
), unit AS (
  SELECT t.taker_id, side.in_sample, t.event_id,
         sum(t.s) AS s, sum(t.notional) AS notional, count(*) AS n
  FROM read_parquet('{trades}') t JOIN side USING (event_id)
  GROUP BY 1, 2, 3
)
SELECT taker_id, in_sample,
       count(*)      AS n_events,
       sum(n)        AS n_trades,
       sum(s)        AS pnl,
       sum(s * s)    AS var_event,
       sum(notional) AS notional
FROM unit GROUP BY 1, 2
"""

DROPPED_SQL = """
WITH ev AS (
  SELECT event_id, min(ts) AS mn, max(ts) AS mx
  FROM read_parquet('{trades}') GROUP BY 1
)
SELECT count(*) FILTER (WHERE mn < {cut} AND mx >= {cut}) AS straddling,
       count(*) AS total FROM ev
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default=os.path.join(ROOT, "data", "pmv1_trades.parquet"))
    ap.add_argument("--cut", default="2025-01-01")
    ap.add_argument("--min-trades", type=int, default=10,
                    help="minimum fills required in EACH half")
    ap.add_argument("--memory-gb", type=int, default=20)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    if not os.path.exists(a.trades):
        sys.exit(f"{a.trades} missing -- run pm/pmv1_prep.py first")
    cut = int(pd.Timestamp(a.cut, tz="UTC").timestamp())

    import duckdb
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{a.memory_gb}GB'")
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute(f"PRAGMA temp_directory='{os.path.join(ROOT, 'data', 'duckdb_tmp')}'")
    straddle, total = con.execute(DROPPED_SQL.format(trades=a.trades, cut=cut)).fetchone()
    d = con.execute(HALVES_SQL.format(trades=a.trades, cut=cut)).df()
    con.close()
    print(f"events {total:,}   dropped for straddling the cut {straddle:,} "
          f"({straddle/total:.1%})")

    ins = d[d.in_sample].set_index("taker_id")
    oos = d[~d.in_sample].set_index("taker_id")
    both = ins.index.intersection(oos.index)
    print(f"cut {a.cut}   accounts in-sample {len(ins):,}   out-of-sample {len(oos):,}   "
          f"in both {len(both):,}")

    def z_of(x):
        sd = np.sqrt(x.var_event.to_numpy())
        return np.divide(x.pnl.to_numpy(), sd, out=np.zeros(len(x)), where=sd > 0)

    ins, oos = ins.loc[both], oos.loc[both]
    m = pd.DataFrame({
        "n_in": ins.n_trades.to_numpy(), "n_out": oos.n_trades.to_numpy(),
        "ev_in": ins.n_events.to_numpy(), "ev_out": oos.n_events.to_numpy(),
        "z_in": z_of(ins), "z_out": z_of(oos),
        "pnl_in": ins.pnl.to_numpy(), "pnl_out": oos.pnl.to_numpy(),
        "not_in": ins.notional.to_numpy(), "not_out": oos.notional.to_numpy(),
    }, index=both)
    m = m[(m.n_in >= a.min_trades) & (m.n_out >= a.min_trades)]
    print(f"accounts with >={a.min_trades} trades in both halves: {len(m):,}")
    print(f"in-sample  trades {m.n_in.sum():,}  notional ${m.not_in.sum():,.0f}  "
          f"PnL ${m.pnl_in.sum():,.0f}")
    print(f"out-sample trades {m.n_out.sum():,}  notional ${m.not_out.sum():,.0f}  "
          f"PnL ${m.pnl_out.sum():,.0f}")

    rho = m.z_in.rank().corr(m.z_out.rank())
    rho_p = m.pnl_in.rank().corr(m.pnl_out.rank())
    print(f"\nSpearman(z_in, z_out)   {rho:+.4f}   "
          f"(2 s.e. under no persistence: +-{2/np.sqrt(len(m)):.4f})")
    print(f"Spearman(pnl_in, pnl_out) {rho_p:+.4f}")

    print(f"\n{'in-sample z decile':<20}{'n':>8}{'mean z_in':>12}{'mean z_out':>12}"
          f"{'median z_out':>14}{'PnL_out $':>16}{'PnL_out/$ notional':>20}")
    dec = pd.qcut(m.z_in, 10, labels=False, duplicates="drop")
    rows = []
    for i in sorted(pd.unique(dec)):
        s = m[dec == i]
        roi = s.pnl_out.sum() / max(s.not_out.sum(), 1)
        rows.append(dict(decile=int(i) + 1, n=len(s), z_in=float(s.z_in.mean()),
                         z_out=float(s.z_out.mean()), pnl_out=float(s.pnl_out.sum()),
                         roi_out=float(roi)))
        print(f"{'D' + str(int(i)+1):<20}{len(s):>8,}{s.z_in.mean():>+12.3f}"
              f"{s.z_out.mean():>+12.3f}{s.z_out.median():>+14.3f}"
              f"{s.pnl_out.sum():>+16,.0f}{roi:>+20.4%}")

    base = (m.z_out > 1.645).mean()
    flag = m.z_in > 1.645
    hit = (m.z_out[flag] > 1.645).mean() if flag.any() else np.nan
    print(f"\nflagged in-sample (z>1.645): {flag.mean():.2%} of accounts "
          f"({flag.sum():,})")
    print(f"of those, flagged out-of-sample: {hit:.2%}   base rate {base:.2%}   "
          f"lift {hit/base if base > 0 else np.nan:.2f}x")
    top, bot = m[dec == dec.max()], m[dec == 0]
    print(f"top decile out-of-sample: PnL ${top.pnl_out.sum():,.0f} on "
          f"${top.not_out.sum():,.0f} notional ({top.pnl_out.sum()/max(top.not_out.sum(),1):+.3%})")
    print(f"bottom decile out-of-sample: PnL ${bot.pnl_out.sum():,.0f} on "
          f"${bot.not_out.sum():,.0f} notional ({bot.pnl_out.sum()/max(bot.not_out.sum(),1):+.3%})")
    print(f"all accounts out-of-sample: {m.pnl_out.sum()/max(m.not_out.sum(),1):+.3%} of notional")

    out = os.path.join(ROOT, "data", f"pmv1_persistence_{a.cut}.json")
    json.dump({"cut": a.cut, "n_accounts": int(len(m)),
               "spearman_z": float(rho), "spearman_pnl": float(rho_p),
               "hit": float(hit), "base": float(base), "deciles": rows,
               "pnl_out_total": float(m.pnl_out.sum()),
               "notional_out_total": float(m.not_out.sum())}, open(out, "w"), indent=1)
    m.to_csv(os.path.join(ROOT, "data", f"pmv1_persistence_{a.cut}.csv"))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
