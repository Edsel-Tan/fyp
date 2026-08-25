#!/usr/bin/env python3
"""Write the portfolio out as a composite-benchmark definition for the tracker.

Emits two baskets, because they answer different questions:

  FYP30  - the live portfolio, selected using data through 2026-08-07. Any
           history the tracker draws before that date is a HINDSIGHT backtest,
           not a track record, and is labelled as such.
  FYP25P - the same rules run at 2025-08-01 on point-in-time data only, so its
           entire plotted history is genuinely out of sample. This is the
           honest comparator; FYP30 is the one to actually buy.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel2 import total_return_panel, load_meta
from factors2 import prep, fundamentals_asof, factor_table, universe
from walkforward2 import screen, score_and_pick

OUT = "/home/aimer/services/finance/composites"
INCEPTION = "2025-03-04"     # start of the tracker's own cached price history


def basket(adj, vol, meta, inc, bal, cf, asof, n):
    funds = fundamentals_asof(inc, bal, cf, pd.Timestamp(asof))
    ft = factor_table(adj, vol, meta, funds, pd.Timestamp(asof))
    u = universe(ft, adj, pd.Timestamp(asof))
    return score_and_pick(screen(u), n), len(u)


def main():
    close, adj, vol = total_return_panel()
    meta, inc, bal, cf, rep = load_meta()
    inc, bal, cf = prep(inc), prep(bal), prep(cf)
    os.makedirs(OUT, exist_ok=True)

    # FYP30 is plotted from the tracker's earliest cached price so it is usable
    # as a benchmark at all; that stretch is hindsight and says so. FYP25P is
    # plotted only from its own formation date, so nothing in it is hindsight.
    specs = [
        ("FYP30", "2026-08-07", 30,
         "FYP quality/value/growth screen, 30 names, equal weight, never rebalanced. "
         "Selected using data through 2026-08-07 -- history plotted before that date is a "
         "hindsight backtest, NOT a track record. Compare FYP25P for an out-of-sample series."),
        ("FYP25P", "2025-08-01", 25,
         "Same screen run at 2025-08-01 on point-in-time data only (income, balance sheet and "
         "cash flow lagged 75 days; no later information used). Its entire history is out of "
         "sample, so this is the honest read on whether the rule works."),
    ]

    for sym, asof, n, note in specs:
        p, nuniv = basket(adj, vol, meta, inc, bal, cf, asof, n)
        w = round(1.0 / len(p), 6)
        d = {
            "symbol": sym,
            "name": f"FYP screened {len(p)}, equal weight, un-rebalanced",
            "inception": INCEPTION if sym == "FYP30" else asof,
            "formed": asof,
            "weights": {s: w for s in p.index},
            "sectors": p["sector"].value_counts().to_dict(),
            "universe_at_formation": int(nuniv),
            "note": note,
        }
        path = os.path.join(OUT, f"{sym.lower()}.json")
        with open(path, "w") as fh:
            json.dump(d, fh, indent=2, sort_keys=False)
        print(f"{sym}: {len(p)} names, {w:.4%} each -> {path}")
        print("   ", ", ".join(p.index))


if __name__ == "__main__":
    main()
