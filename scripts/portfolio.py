#!/usr/bin/env python3
"""Build the buy-and-hold portfolio.

The screen is fixed in advance and is NOT fitted to this 2-year tape (see
walkforward.py for why fitting it would be a mistake). It selects for the things
that let a position be held for years without maintenance:

  survivability  -- profitable, cash-generative, not dependent on refinancing
  durability     -- real gross margin, stable/improving operating margin
  growth         -- revenue and earnings still compounding
  valuation      -- not paying a multiple that needs perfection
  liquidity      -- large enough that it will not need replacing

Then equal-weights ~25 names with a hard sector cap.

Usage: portfolio.py [--asof YYYY-MM-DD] [--n 25] [--pit]
  --pit  restrict to point-in-time income-statement data only (for backtesting a
         formation date in the past, where the balance sheet snapshot would leak)
"""
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_panel import build
from factors import factor_table, universe, zs
from walkforward import prep_fh

MIN_MCAP = 2e9
MIN_DVOL = 2e7
SECTOR_CAP = 4          # max names per sector


def add_balance_sheet(u, fl, asof):
    """Balance-sheet STOCK items from the fundamentals_latest snapshot.

    Two traps in this table, both verified against the tape:

    1. period_type is mostly a single quarter (median fl.revenue / TTM revenue
       = 0.25), so every FLOW in it -- net_income, ebitda, free_cash_flow -- is
       a quarterly figure for most names. Flows therefore come from the TTM
       aggregates built off fundamentals_history instead; only balance-sheet
       stocks are taken from here.
    2. ~26 symbols report the balance sheet in thousands while the income
       statement is in dollars (ALGN: equity 4.2e6 against net income 1.1e8).
       Detected by scale against TTM revenue and repaired, or dropped.
    """
    b = fl.copy()
    b["rd"] = pd.to_datetime(b["report_date"], errors="coerce")
    b = b[b["rd"] <= pd.Timestamp(asof)].set_index("symbol")
    b = b[~b.index.duplicated(keep="last")]
    stocks = ["total_equity", "total_debt", "cash_and_equivalents", "total_assets"]
    j = u.join(b[stocks + ["free_cash_flow", "stock_repurchases",
                          "period_type"]].add_prefix("bs_"), how="left")

    # --- unit repair -------------------------------------------------------
    scale = j["bs_total_assets"] / j["revenue_ttm"].where(j["revenue_ttm"] > 0)
    bad = scale < 0.01
    for c in stocks:
        j.loc[bad, "bs_" + c] = j.loc[bad, "bs_" + c] * 1000.0
    j["unit_fixed"] = bad.fillna(False)
    # anything still implausible after repair is not trustworthy -- drop it
    rescale = j["bs_total_assets"] / j["revenue_ttm"].where(j["revenue_ttm"] > 0)
    j = j[~(rescale < 0.05)]

    # --- ratios: flows from TTM history, stocks from the snapshot ----------
    j["net_debt"] = j["bs_total_debt"] - j["bs_cash_and_equivalents"]
    j["nd_ebitda"] = j["net_debt"] / j["ebitda_ttm"].where(j["ebitda_ttm"] > 0)
    j["roe"] = j["net_income_ttm"] / j["bs_total_equity"].where(j["bs_total_equity"] > 0)
    j["roa"] = j["net_income_ttm"] / j["bs_total_assets"].where(j["bs_total_assets"] > 0)
    j["equity_ratio"] = j["bs_total_equity"] / j["bs_total_assets"]
    # FCF sign is period-independent, so use it as a gate but not as a yield
    j["fcf_pos"] = j["bs_free_cash_flow"] > 0
    j["ey"] = j["net_income_ttm"] / j["mcap"]

    # Buyback yield. dividends_paid is 100% NULL in this database and close is
    # not dividend-adjusted, so cash returned as dividends is invisible here and
    # in any case cannot compound in an un-rebalanced account without a trade.
    # Repurchases compound on their own -- they are the shareholder return that
    # actually suits this mandate. Negative stock_repurchases = cash paid out to
    # buy shares back; positive = net issuance, i.e. the holder is being diluted.
    # A single quarter annualised x4 is a lumpy estimate: PAYC bought back
    # $1.05bn in one quarter against $156m of quarterly earnings, which scales to
    # an absurd 60% of its market cap. Cap the run-rate at TTM net income, so
    # this measures the share of earnings actually returned via buyback and can
    # never exceed the earnings yield.
    mult = j["bs_period_type"].map({"TTM": 1.0, "FY": 1.0}).fillna(4.0)  # Qx -> annualise
    gross_bb = (-j["bs_stock_repurchases"] * mult).clip(lower=0)
    j["buyback_yield"] = np.minimum(gross_bb, j["net_income_ttm"].clip(lower=0)) / j["mcap"]
    return j


def screen(u, pit):
    """Hard survivability gates -- everything here is a veto, not a score."""
    ok = (
        (u["mcap"] >= MIN_MCAP)
        & (u["mdvol"] >= MIN_DVOL)
        & (u["net_income_ttm"] > 0)          # actually earns money
        & (u["operating_income_ttm"] > 0)
        & (u["om"] > 0.05)                   # real operating margin
        & (u["gm"] > 0.20)                   # something to defend
        & (u["rev_g"] > 0.02)                # still growing
        & (u["om_chg"] > -0.05)              # margins not collapsing
        & (u["dd1y"] > -0.60)                # not a falling knife
    )
    if not pit:
        ok = ok & (
            u["fcf_pos"]                     # self-funding
            & (u["roe"] > 0.10)              # earns its keep on shareholder capital
            & (u["roe"] < 2.0)               # absurd ROE means broken/negative equity
            & (u["equity_ratio"] > 0.15)     # not levered to the eyeballs
            & ~(u["nd_ebitda"] > 3.0)        # survives without refinancing
            & (u["ey"] > 0.02)               # not priced for perfection
        )
    return u[ok.fillna(False)].copy()


def score(c, pit):
    """Composite of quality, growth and valuation. Equal weight on each pillar --
    no fitted coefficients, because 2 years of tape cannot identify them."""
    quality = zs(c["om"]) + zs(c["gm"]) + (0 if pit else zs(c["roe"]) + zs(c["roa"]))
    growth = zs(c["rev_g"]) + zs(c["eps_g"].fillna(c["rev_g"])) + zs(c["om_chg"])
    value = zs(c["sales_y"]) + zs(c["ey"] if "ey" in c else c["sales_y"])
    # NOT a low-vol tilt: that pillar selects utilities/REITs, whose return is
    # mostly dividends -- invisible in this data and unable to compound without
    # a trade. Reward internal compounding and penalise dilution instead.
    pillars = [("quality", quality), ("growth", growth), ("value", value)]
    if not pit:
        # only available live: fundamentals_history carries no cash-flow items,
        # so this pillar cannot be evaluated at a past formation date at all.
        pillars.append(("compounding", zs(c["buyback_yield"])))
    for name, s in pillars:
        c[name] = s / s.std()
    return c


PILLARS = ["quality", "growth", "value", "compounding"]
PILLARS_PIT = ["quality", "growth", "value"]


def pick(c, n, cap=SECTOR_CAP, cols=None):
    c = c.copy()
    cols = cols or [x for x in PILLARS if x in c.columns]
    c["total"] = c[cols].mean(axis=1)
    c = c.sort_values("total", ascending=False)
    out, used = [], {}
    for sym, row in c.iterrows():
        s = row["sector"]
        if used.get(s, 0) >= cap:
            continue
        used[s] = used.get(s, 0) + 1
        out.append(sym)
        if len(out) == n:
            break
    return c.loc[out]


def build_portfolio(asof, n=25, pit=False, data=None):
    close, dvol, meta, fl, fh = data if data else build()
    if not isinstance(fh.get("avail", None), pd.Series):
        fh = prep_fh(fh)
    f = factor_table(close, dvol, meta, fh, asof)
    u = universe(f, close, asof)
    if not pit:
        u = add_balance_sheet(u, fl, asof)
    c = screen(u, pit)
    c = score(c, pit)
    return pick(c, n, cols=PILLARS_PIT if pit else PILLARS), c, (close, dvol, meta, fl, fh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-08-07")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--pit", action="store_true")
    a = ap.parse_args()

    port, cand, _ = build_portfolio(pd.Timestamp(a.asof), a.n, a.pit)
    print(f"as of {a.asof}   candidates passing the screen: {len(cand)}   holding {len(port)}")
    cols = ["sector", "price", "mcap", "om", "gm", "rev_g", "total"]
    if not a.pit:
        cols = ["sector", "price", "mcap", "roe", "ey", "nd_ebitda",
                "om", "rev_g", "buyback_yield", "total"]
    show = port[cols].copy()
    show["mcap"] = (show["mcap"] / 1e9).round(1)
    print(show.round(3).to_string())
    print("\nsector mix:")
    print(port["sector"].value_counts().to_string())
