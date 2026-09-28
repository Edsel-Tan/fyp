#!/usr/bin/env python3
"""Decompose the price-baseline book. Is +9 pp a signal or a structure? ([16])

`pmv1_text_signal.py` reports that M1 -- price state and category, no text -- runs
a long-short book worth several probability points per share net of execution.
That is a large number on a venue where RESULTS.md sec.3.5 found the taker side of
the imbalance signal 4.2x under water, and it should not be believed until the
obvious structural explanations are ruled out. This script rules them out, or
fails to.

Four decompositions of the same book:

  by neg_risk   Negative-risk multi-outcome events list each candidate as its own
                market, and their p_event values are not designed to sum to one
                across the event (DATA.md sec.9). A book that shorts the favourite
                and longs the field inside one such event is harvesting that
                construction, not a mispricing. If the edge lives only in
                neg_risk = true, it is an artifact of the data layout.
  by side       Shorting overpriced favourites and buying underpriced longshots are
                different trades with different capacity and different borrow
                assumptions. On a binary contract a short is a long of the other
                leg, so both are executable -- but they should be reported apart.
  by price      Where on the calibration curve the money is. sec.3's curve says
                favourites at p in [0.6,0.9] are overpriced ~4 pp.
  by event      Concentration. With ~700 events in a test month, a handful of
                large resolutions can carry the mean, and the event-clustered t
                will not catch it if the cluster itself is the bet.

    python pm/pmv1_text_book.py --folds 10
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pmv1_text_signal as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="nl")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--binary-only", action="store_true")
    ap.add_argument("--same-bar", action="store_true",
                    help="execute in the signal's own bar -- the look-ahead, quoted")
    a = ap.parse_args()

    df = S.load(a.kind, "term")
    df = S.apply_null(df, "none", 0)
    if a.binary_only:
        df = df[~df.neg_risk.astype(bool)].reset_index(drop=True)
    cats = df.cat.value_counts().head(20).index.tolist()
    Xp_all = S.price_block(df, cats)
    y, w, ev = df.y.values, df.w.values, df.event_id.values

    parts = []
    for name, tr, va, te in S.make_folds(df, "walk", a.folds, 0):
        if va.sum() < 200:
            continue
        Xp = S.zscore(Xp_all, tr)
        _, al, _, m = S.best_ridge(lambda g: Xp, y, w, tr, va, ev[va], [0.0])
        pred = m.predict(Xp[te])
        d = df[te].copy()
        hi, lo = np.quantile(pred, .8), np.quantile(pred, .2)
        side = np.where(pred >= hi, 1.0, np.where(pred <= lo, -1.0, 0.0))
        k = side != 0
        dd, s = d[k], side[k]
        if a.same_bar:
            buy, sell = dd.vwap_buy, dd.vwap_sell
        else:
            buy, sell = dd.vwap_buy_next, dd.vwap_sell_next
        ok = (buy.notna() & sell.notna()).values
        dd, s = dd[ok], s[ok]
        buy, sell = buy[ok].values, sell[ok].values
        ref = (buy + sell) / 2.0        # spread-free benchmark at the same timing
        half = np.maximum((buy - sell) / 2.0, 0.0)   # never paid to cross
        ex = ref + np.where(s > 0, half, -half)
        parts.append(pd.DataFrame({
            "fold": name, "gross": (s * (dd.win_event - ref)).values,
            "net": (s * (dd.win_event - ex)).values, "side": s, "p": dd.p.values,
            "neg": dd.neg_risk.astype(bool).values, "cat": dd.cat.values,
            "vol": dd.gross_day.values, "event_id": dd.event_id.values,
            "w": dd.w.values}))
    b = pd.concat(parts, ignore_index=True)
    pd.set_option("display.width", 200)

    def tab(key, label):
        out = []
        for k, g in b.groupby(key, observed=True):
            st = S.event_stats(g.net, g.event_id, g.w)
            sg = S.event_stats(g.gross, g.event_id, g.w)
            out.append(dict(**{label: k}, trips=len(g), events=st["n_events"],
                            gross_pp=sg["mean_pp"], net_pp=st["mean_pp"], t_net=st["t"]))
        return pd.DataFrame(out).round(4)

    all_ = S.event_stats(b.net, b.event_id, b.w)
    allg = S.event_stats(b.gross, b.event_id, b.w)
    print(f"M1 book, {len(b):,} trips over {all_['n_events']:,} events"
          f"{'  [binary markets only]' if a.binary_only else ''}")
    print(f"  gross {allg['mean_pp']:+.4f} pp (t={allg['t']:+.2f})   "
          f"net {all_['mean_pp']:+.4f} pp (t={all_['t']:+.2f})\n")

    b["side_l"] = np.where(b.side > 0, "long", "short")
    b["pbin"] = pd.cut(b.p, [0, .05, .15, .35, .65, .85, 1.0])
    b["volq"] = pd.qcut(b.vol, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    for key, lab in [("neg", "neg_risk"), ("side_l", "side"), ("pbin", "price"),
                     ("volq", "volume"), ("fold", "fold")]:
        print(f"--- by {lab} ---")
        print(tab(key, lab).to_string(index=False)); print()

    # concentration: how much of the total comes from the largest events?
    g = S.event_mean(b.net, b.event_id, b.w).sort_values()
    tot = g.sum()
    print("concentration of net PnL across events:")
    for n in (1, 5, 10, 25):
        print(f"  top {n:>2} events by |contribution| = "
              f"{100*g.abs().nlargest(n).sum()/g.abs().sum():.1f}% of gross |PnL|;  "
              f"dropping them leaves mean {100*g.drop(g.abs().nlargest(n).index).mean():+.4f} pp")
    json.dump(dict(gross_pp=allg["mean_pp"], net_pp=all_["mean_pp"], t_net=all_["t"],
                   trips=len(b), events=all_["n_events"], binary_only=a.binary_only),
              open(os.path.join(S.NLP, "book_m1" + ("_binary" if a.binary_only else "")
                                + ("_samebar" if a.same_bar else "") + ".json"), "w"),
              indent=2, default=float)


if __name__ == "__main__":
    main()
