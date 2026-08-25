#!/usr/bin/env python3
"""Point-in-time factors on the deep, dividend-adjusted panel.

Everything here is as-of `asof`: income, balance sheet and cash flow all come
from quarterly history lagged by a reporting delay, and market cap is derived
from the price on the day rather than a present-day snapshot. Nothing reads a
value that would not have been on a screen at the formation date.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel2 import total_return_panel, load_meta

LAG_DAYS = 75          # reporting delay applied to period_end_date
FLOWS = ["revenue", "gross_profit", "operating_income", "ebitda", "net_income", "eps",
         "operating_cash_flow", "free_cash_flow", "capex", "stock_repurchases"]
STOCKS = ["total_assets", "total_liabilities", "total_equity",
          "cash_and_equivalents", "total_debt"]


def prep(df):
    df = df[df["period_type"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
    df["avail"] = df["pe"] + pd.Timedelta(days=LAG_DAYS)
    # a symbol occasionally carries two rows for one period end (restatement or
    # a duplicated filing); keep the last and make (symbol, pe) a real key
    return (df.sort_values(["symbol", "pe"])
              .drop_duplicates(subset=["symbol", "pe"], keep="last"))


def fundamentals_asof(inc, bal, cf, asof):
    """TTM flows and latest balance-sheet stocks known at `asof`."""
    asof = pd.Timestamp(asof)

    cfa = cf[cf["avail"] <= asof].drop(
        columns=["period_end_date", "period_type", "avail"], errors="ignore")
    flows = inc[inc["avail"] <= asof].merge(cfa, on=["symbol", "pe"], how="left")
    flows = flows.sort_values(["symbol", "pe"])

    have = [c for c in FLOWS if c in flows.columns]
    g = flows.groupby("symbol")
    for c in have:
        flows[c + "_ttm"] = g[c].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    cur = flows.groupby("symbol").tail(1).set_index("symbol")
    prev = flows.groupby("symbol").nth(-5)
    prev = prev.set_index("symbol") if "symbol" in prev.columns else prev

    out = cur[[c + "_ttm" for c in have]].copy()
    out = out.join(prev[[c + "_ttm" for c in have]].add_suffix("_p"), how="left")

    b = bal[bal["avail"] <= asof]
    b = b.groupby("symbol").tail(1).set_index("symbol")
    out = out.join(b[[c for c in STOCKS if c in b.columns]], how="left")
    out["fund_age_d"] = (asof - cur["pe"]).dt.days
    return out


def factor_table(adj, vol, meta, funds, asof):
    asof = pd.Timestamp(asof)
    px = adj.loc[:asof]
    dv = vol.loc[:asof]
    lb = min(252, len(px))
    if lb < 200:
        raise ValueError("need >=200 trading days at formation")

    last = px.iloc[-1]
    ret = np.log(px).diff()

    f = pd.DataFrame({
        "price": last,
        "mom12_1": px.iloc[-22] / px.iloc[-lb] - 1,
        "mom6": px.iloc[-1] / px.iloc[-126] - 1,
        "vol": ret.iloc[-lb:].std() * np.sqrt(252),
        "dd1y": (px.iloc[-lb:] / px.iloc[-lb:].cummax() - 1).min(),
        "trend": (px.iloc[-lb:] > px.iloc[-lb:].rolling(50).mean()).mean(),
        "mdvol": (dv.iloc[-lb:] * px.iloc[-lb:]).median(),
    })
    f = f.join(funds, how="left")

    # ---- point-in-time market cap ------------------------------------------
    # shares are backed out of the present-day cap at each symbol's own last
    # traded price, then repriced at the formation date. Deriving them against
    # the last date of the panel instead would NaN out every delisted name and
    # silently rebuild the survivorship bias.
    m = meta.set_index("symbol")
    last_traded = adj.apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    shares = m["market_cap"] / last_traded.reindex(m.index)
    f["shares"] = shares.reindex(f.index)
    f["mcap"] = f["shares"] * f["price"]

    f["rev_g"] = f["revenue_ttm"] / f["revenue_ttm_p"] - 1
    f["eps_g"] = np.where(f["eps_ttm_p"] > 0, f["eps_ttm"] / f["eps_ttm_p"] - 1, np.nan)
    f["gm"] = f["gross_profit_ttm"] / f["revenue_ttm"]
    f["om"] = f["operating_income_ttm"] / f["revenue_ttm"]
    f["om_chg"] = f["om"] - f["operating_income_ttm_p"] / f["revenue_ttm_p"]
    f["fcf_margin"] = f["free_cash_flow_ttm"] / f["revenue_ttm"]
    f["roe"] = f["net_income_ttm"] / f["total_equity"].where(f["total_equity"] > 0)
    f["roa"] = f["net_income_ttm"] / f["total_assets"].where(f["total_assets"] > 0)
    f["equity_ratio"] = f["total_equity"] / f["total_assets"]
    f["net_debt"] = f["total_debt"] - f["cash_and_equivalents"]
    f["nd_ebitda"] = f["net_debt"] / f["ebitda_ttm"].where(f["ebitda_ttm"] > 0)
    f["ey"] = f["net_income_ttm"] / f["mcap"]
    f["fcf_yield"] = f["free_cash_flow_ttm"] / f["mcap"]
    f["sales_y"] = f["revenue_ttm"] / f["mcap"]
    # `stock_repurchases` only exists on the fundamentals_latest snapshot, never
    # in the quarterly history, so buyback yield cannot be formed at a past date.
    # Cash conversion and accruals carry the same idea -- earnings that are real
    # and get reinvested rather than accrued -- and are available point-in-time.
    if "stock_repurchases_ttm" in f.columns:
        bb = (-f["stock_repurchases_ttm"]).clip(lower=0)
        f["buyback_yield"] = np.minimum(bb, f["net_income_ttm"].clip(lower=0)) / f["mcap"]
    else:
        f["buyback_yield"] = np.nan
    f["accruals"] = (f["net_income_ttm"] - f["operating_cash_flow_ttm"]) / f["total_assets"]
    f["cash_conv"] = f["free_cash_flow_ttm"] / f["net_income_ttm"].where(f["net_income_ttm"] > 0)

    # unit mismatch: a handful of filers report the balance sheet in thousands
    # against an income statement in dollars
    scale = f["total_assets"] / f["revenue_ttm"].where(f["revenue_ttm"] > 0)
    f["unit_suspect"] = scale < 0.05

    return f.join(m[["sector", "industry", "exchange", "country", "active", "name"]])


def universe(f, adj, asof, min_dvol=5e6, min_price=5.0, min_mcap=3e8, need=200):
    asof = pd.Timestamp(asof)
    px = adj.loc[:asof]
    full = px.iloc[-need:].notna().all()
    ok = (
        full.reindex(f.index).fillna(False)
        & (f["price"] >= min_price)
        & (f["mdvol"] >= min_dvol)
        & f["sector"].notna()
        & (f["revenue_ttm"] > 0)
        & ~(f["mcap"] < min_mcap)
        & f["country"].isin(["US"])
        & ~f["unit_suspect"].fillna(False)
        & (f["fund_age_d"] < 400)          # drop stale filers
    )
    return f[ok.fillna(False)].copy()


def zs(s):
    return pd.to_numeric(s, errors="coerce").rank(pct=True) - 0.5
