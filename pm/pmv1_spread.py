#!/usr/bin/env python3
"""Is the large-trade imbalance signal tradeable once the spread is paid? ([14], [16])

RESULTS.md sec.3.3 establishes that large-trade order imbalance predicts the next
bar (beta = +0.0033 at 1h, t = +13.3) and that the effect lives entirely in the top
liquidity quintile (beta = +0.0056, t = +8.92). sec.10 item 4 says what is missing:
a cost. It also says why the cost is awkward here -- `fee_usdc` is identically zero
across the whole archive, because Polymarket v1 charged no explicit taker fee, so
the only cost is the spread and it has to be *estimated*.

Estimating it needs trade prices, which the compacted tables in pmv1_prep.py drop.
So this script does its own pass over the cleaned layers and keeps, per bar, the
volume-weighted price at which takers bought and the one at which takers sold.
Because `D` is ground truth from settlement rather than a tick rule, the two sides
are correctly separated, and their difference

    spread_p = vwap(p_event | D=+1) - vwap(p_event | D=-1)

is an effective spread in probability units.

The backtest then prices the round trip *at those VWAPs* rather than at the bar's
last print. That matters for a reason sec.3.3 already names: the last print carries
bid-ask bounce, which pushes the next return against the signal, so a backtest that
enters at the last print and then subtracts a spread charges the crossing twice.
Trading at the side a taker must actually hit removes the bounce and the spread
together, and needs no cost model at all.

  signal known at the close of bar h; position opened in bar h+1, closed in h+2
    long  : buy at vwap_buy(h+1),  sell at vwap_sell(h+2), capital = entry price
    short : sell at vwap_sell(h+1), buy at vwap_buy(h+2),  capital = 1 - entry price
  the same trip priced mid-to-mid is the gross benchmark, and the difference is
  what the two crossings cost.

    python pm/pmv1_spread.py --freq 1h
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
PMV1 = os.path.join(ROOT, "data", "pmv1")
sys.path.insert(0, HERE)

# Same bar construction as pmv1_prep.BARS -- same large-trade threshold, same
# min-volume filter -- plus the two taker-side VWAPs the cost needs.
BARS = """
COPY (
  WITH src AS (
    SELECT condition_id, category_refined, block_timestamp, p_event, D, usdc_amount
    FROM read_parquet({glob}, union_by_name=true)
    WHERE p_event > 0 AND p_event < 1 AND usdc_amount > 0 AND D IS NOT NULL
  ), mkt AS (
    SELECT condition_id, quantile_cont(usdc_amount, {large_q}) AS thr
    FROM src GROUP BY 1 HAVING sum(usdc_amount) >= {min_volume}
  ), tagged AS (
    SELECT s.*, (s.block_timestamp / {step})::BIGINT AS bar,
           s.usdc_amount >= m.thr AS is_large
    FROM src s JOIN mkt m USING (condition_id)
  )
  SELECT (hash(condition_id) >> 1)::BIGINT AS market_id, bar,
         any_value(category_refined) AS cat,
         sum(usdc_amount) AS gross,
         arg_max(p_event, block_timestamp) AS p,
         sum(CASE WHEN is_large THEN D * usdc_amount ELSE 0 END) AS sgn_large,
         sum(CASE WHEN is_large THEN usdc_amount ELSE 0 END)     AS abs_large,
         sum(CASE WHEN NOT is_large THEN D * usdc_amount ELSE 0 END) AS sgn_small,
         sum(CASE WHEN NOT is_large THEN usdc_amount ELSE 0 END)     AS abs_small,
         sum(CASE WHEN D = 1 THEN p_event * usdc_amount END)
           / nullif(sum(CASE WHEN D = 1 THEN usdc_amount END), 0)   AS vwap_buy,
         sum(CASE WHEN D = -1 THEN p_event * usdc_amount END)
           / nullif(sum(CASE WHEN D = -1 THEN usdc_amount END), 0)  AS vwap_sell
  FROM tagged GROUP BY 1, 2
) TO '{out}' (FORMAT parquet, COMPRESSION zstd)
"""


def build(path, freq, large_q, min_volume, memory_gb, threads):
    import duckdb
    globs = [os.path.join(PMV1, L, "*.parquet")
             for L in ("daily_aligned", "daily_aligned_multi")
             if glob.glob(os.path.join(PMV1, L, "*.parquet"))]
    if not globs:
        sys.exit(f"no parquet under {PMV1}: run pm/pmv1_pull.py first")
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB'")
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"PRAGMA temp_directory='{os.path.join(ROOT, 'data', 'duckdb_tmp')}'")
    print(f"building {path} ...", flush=True)
    con.execute(BARS.format(glob=str(globs), large_q=large_q, min_volume=min_volume,
                            step=int(pd.Timedelta(freq).total_seconds()), out=path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="1h")
    ap.add_argument("--large-q", type=float, default=0.90)
    ap.add_argument("--min-volume", type=float, default=100_000)
    ap.add_argument("--memory-gb", type=int, default=24)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    path = os.path.join(ROOT, "data", f"pmv1_spreadbars_{a.freq}.parquet")
    if a.rebuild or not os.path.exists(path):
        build(path, a.freq, a.large_q, a.min_volume, a.memory_gb, a.threads)

    b = pd.read_parquet(path).sort_values(["market_id", "bar"], kind="stable")
    two = b.vwap_buy.notna() & b.vwap_sell.notna()
    print(f"freq {a.freq}   bars {len(b):,}   markets {b.market_id.nunique():,}   "
          f"both sides trade in {two.mean():.1%} of bars")

    # --- the spread itself -----------------------------------------------------
    s = b[two]
    q = pd.qcut(s.gross, 5, labels=False, duplicates="drop")
    print("\n=== effective spread by bar-liquidity quintile ===")
    print(f"{'quintile':<10}{'bars':>11}{'median gross':>14}{'median p':>10}"
          f"{'median spread':>15}{'mean spread':>13}{'as % of price':>15}")
    sp = {}
    for i in sorted(pd.unique(q.dropna())):
        x = s[q == i]
        sp[f"Q{int(i)+1}"] = dict(n=int(len(x)), median_gross=float(x.gross.median()),
                                  median_p=float(x.p.median()),
                                  median_spread=float(x.spread.median()) if False else
                                  float((x.vwap_buy - x.vwap_sell).median()),
                                  mean_spread=float((x.vwap_buy - x.vwap_sell).mean()))
        d = x.vwap_buy - x.vwap_sell
        print(f"{'Q'+str(int(i)+1):<10}{len(x):>11,}{x.gross.median():>14,.0f}"
              f"{x.p.median():>10.3f}{d.median():>15.5f}{d.mean():>13.5f}"
              f"{(d / x.p.clip(0.01, 0.99)).median():>15.2%}")

    # --- the round trip, priced at prices a taker actually got -----------------
    g = b.groupby("market_id", sort=False)
    b["oi_large"] = b.sgn_large / (b.abs_large + 1e-9)
    b["mid"] = (b.vwap_buy + b.vwap_sell) / 2
    HOLDS = [1, 2, 4, 8, 24]
    for k in (1,) + tuple(1 + h for h in HOLDS):
        for c in ("vwap_buy", "vwap_sell", "mid", "bar", "gross"):
            b[f"{c}_{k}"] = g[c].shift(-k)

    def trip(hold):
        """Open in bar h+1, close in bar h+1+hold, at the side a taker must hit.

        The cost is one round trip however long the hold, so if the edge keeps
        accumulating the spread is amortised -- this is the only way the signal
        can become tradeable, and it is the test sec.10 item 4 asks for.
        """
        k = 1 + hold
        x = b[(b[f"bar_1"] == b.bar + 1) & (b[f"bar_{k}"] == b.bar + k)]
        x = x.dropna(subset=["oi_large", "vwap_buy_1", "vwap_sell_1",
                             f"vwap_buy_{k}", f"vwap_sell_{k}", "mid_1", f"mid_{k}"])
        x = x[np.sign(x.oi_large) != 0].copy()
        sg = np.sign(x.oi_large.to_numpy())
        lo = sg > 0
        entry = np.where(lo, x.vwap_buy_1, x.vwap_sell_1)
        ex = np.where(lo, x[f"vwap_sell_{k}"], x[f"vwap_buy_{k}"])
        x["pnl_net"] = np.where(lo, ex - entry, entry - ex)
        x["pnl_gross"] = sg * (x[f"mid_{k}"].to_numpy() - x.mid_1.to_numpy())
        x["capital"] = np.where(lo, entry, 1 - entry)
        # cost = the half-spread paid entering + the half-spread paid leaving, so a
        # hold that lengthens does not lengthen the cost unless the exit bar is a
        # wider one. Reporting the two separately keeps that checkable.
        x["hs_in"] = np.where(lo, entry - x.mid_1, x.mid_1 - entry)
        x["hs_out"] = np.where(lo, x[f"mid_{k}"] - ex, ex - x[f"mid_{k}"])
        return x[x.capital > 0.01]

    def line(name, x, res):
        if len(x) < 500:
            return
        gr, nt = x.pnl_gross.to_numpy(), x.pnl_net.to_numpy()
        tg = gr.mean() / (gr.std(ddof=1) / np.sqrt(len(gr)))
        tn = nt.mean() / (nt.std(ddof=1) / np.sqrt(len(nt)))
        # value-weighted, not a mean of ratios: capital can be 0.01, and a mean of
        # pnl/capital is then decided by a handful of penny contracts
        roi_g = gr.sum() / x.capital.sum()
        roi_n = nt.sum() / x.capital.sum()
        res[name] = dict(n=int(len(x)), gross_pp=float(gr.mean() * 100),
                         net_pp=float(nt.mean() * 100),
                         cost_pp=float((gr.mean() - nt.mean()) * 100),
                         roi_gross=float(roi_g), roi_net=float(roi_n),
                         t_gross=float(tg), t_net=float(tn),
                         median_gross_notional=float(x.gross_1.median()),
                         hs_in=float(x.hs_in.mean() * 100),
                         hs_out=float(x.hs_out.mean() * 100))
        print(f"{name:<22}{len(x):>11,}{gr.mean()*100:>+10.4f}{nt.mean()*100:>+10.4f}"
              f"{(gr.mean()-nt.mean())*100:>9.4f}{x.hs_in.mean()*100:>8.4f}"
              f"{x.hs_out.mean()*100:>9.4f}{roi_n:>+9.3%}{tg:>+10.2f}{tn:>+9.2f}")

    HDR2 = (f"{'subset':<22}{'trips':>11}{'gross pp':>10}{'net pp':>10}{'cost pp':>9}"
            f"{'half in':>8}{'half out':>9}{'net ROI':>9}{'t(gross)':>10}{'t(net)':>9}")
    res = {}
    one = trip(1)
    qb = pd.qcut(one.gross_1, 5, labels=False, duplicates="drop")
    print("\n=== sec.3.3's signal as a one-bar round trip at taker prices ===")
    print(HDR2)
    line("all trips", one, res)
    for i in sorted(pd.unique(qb.dropna())):
        line(f"liquidity Q{int(i)+1}", one[qb == i], res)
    print("pp = probability points per share (1 pp = 1 cent on a $1 contract);  "
          "ROI is\nvalue-weighted: total PnL over total capital at risk.")

    print("\n=== can a longer hold amortise the one round trip? (top quintile only) ===")
    print(HDR2)
    hold_res = {}
    for h in HOLDS:
        x = trip(h)
        thr = x.gross_1.quantile(0.8)
        line(f"hold {h}h", x[x.gross_1 >= thr], hold_res)

    print("\n=== and does a stronger signal help? (top quintile, one-bar hold) ===")
    print(HDR2)
    strong = {}
    top = one[qb == qb.max()]
    for lab, m in (("|oi| = 1 (one-sided)", top.oi_large.abs() >= 0.999),
                   ("|oi| > 0.5", top.oi_large.abs() > 0.5),
                   ("|oi| <= 0.5", top.oi_large.abs() <= 0.5)):
        line(lab, top[m], strong)

    top = one[qb == qb.max()]
    gpp, npp = top.pnl_gross.mean() * 100, top.pnl_net.mean() * 100
    print(f"\ntop liquidity quintile: gross {gpp:+.4f} pp/share, the two crossings cost "
          f"{gpp - npp:.4f} pp,\ni.e. {(gpp - npp)/max(abs(gpp), 1e-12):.1f}x the gross edge. "
          f"Net {npp:+.4f} pp/share.")
    # The counterparty of this round trip is passive on both legs, so its PnL is the
    # negative of the taker's net: it collects the two half-spreads and pays the
    # adverse selection the signal measures. That is the same number read the other way.
    print(f"whoever takes the other side of it is passive on both legs and therefore "
          f"earns\n{-npp:+.4f} pp/share: the two half-spreads "
          f"({top.hs_in.mean()*100 + top.hs_out.mean()*100:.4f} pp) less the "
          f"{gpp:.4f} pp of adverse\nselection the signal is measuring.")

    out = os.path.join(ROOT, "data", f"pmv1_spread_{a.freq}.json")
    json.dump({"freq": a.freq, "n_bars": int(len(b)), "two_sided": float(two.mean()),
               "spread_by_quintile": sp, "roundtrip": res, "holds": hold_res,
               "signal_strength": strong}, open(out, "w"), indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
