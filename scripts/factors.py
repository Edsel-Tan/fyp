#!/usr/bin/env python3
"""Point-in-time factor construction + universe screen."""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_panel import build

LAG_DAYS = 75          # income-statement reporting lag applied to period_end_date
MIN_DVOL = 5e6         # median daily $ volume over trailing year
MIN_PRICE = 5.0


def _ttm(fh, asof):
    """TTM income-statement aggregates known as of `asof` (period_end + LAG_DAYS)."""
    f = fh[fh["avail"] <= asof].copy()
    f = f.sort_values(["symbol", "pe"])
    g = f.groupby("symbol")
    out = {}
    for col in ["revenue", "gross_profit", "operating_income", "ebitda", "net_income", "eps"]:
        ttm = g[col].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
        f[col + "_ttm"] = ttm
    last = f.groupby("symbol").tail(1).set_index("symbol")
    # year-ago TTM: the TTM value 4 quarters earlier
    prev = f.groupby("symbol").nth(-5)
    prev = prev.set_index("symbol") if "symbol" in prev.columns else prev
    return last, prev


def factor_table(close, dvol, meta, fh, asof):
    """All factors computed using only information available at `asof`."""
    asof = pd.Timestamp(asof)
    px = close.loc[:asof]
    dv = dvol.loc[:asof]
    if len(px) < 200:
        raise ValueError("need >=200 trading days of history at formation")
    lb = min(252, len(px))              # the tape is only 2y long; use what exists

    ret = np.log(px).diff()
    last = px.iloc[-1]

    # ---- price-based -------------------------------------------------------
    mom12_1 = px.iloc[-22] / px.iloc[-lb] - 1           # 12m return skipping last month
    mom6 = px.iloc[-1] / px.iloc[-126] - 1
    vol = ret.iloc[-lb:].std() * np.sqrt(252)
    dd = (px.iloc[-lb:] / px.iloc[-lb:].cummax() - 1).min()     # worst drawdown in past yr
    # trend consistency: fraction of days spent above the 50d average
    above = (px.iloc[-lb:] > px.iloc[-lb:].rolling(50).mean()).mean()

    f = pd.DataFrame({
        "price": last, "mom12_1": mom12_1, "mom6": mom6, "vol": vol,
        "dd1y": dd, "trend": above, "mdvol": dv.iloc[-lb:].median(),
    })

    # ---- fundamentals (point in time) --------------------------------------
    cur, prev = _ttm(fh, asof)
    f = f.join(cur[[c for c in cur.columns if c.endswith("_ttm")]], how="left")
    f = f.join(prev[[c for c in prev.columns if c.endswith("_ttm")]].add_suffix("_p"), how="left")

    f["rev_g"] = f["revenue_ttm"] / f["revenue_ttm_p"] - 1
    f["eps_g"] = np.where(f["eps_ttm_p"] > 0, f["eps_ttm"] / f["eps_ttm_p"] - 1, np.nan)
    f["gm"] = f["gross_profit_ttm"] / f["revenue_ttm"]
    f["om"] = f["operating_income_ttm"] / f["revenue_ttm"]
    f["nm"] = f["net_income_ttm"] / f["revenue_ttm"]
    f["om_chg"] = f["om"] - f["operating_income_ttm_p"] / f["revenue_ttm_p"]

    # Point-in-time market cap. meta.market_cap is a snapshot taken at the pull
    # date, so back out shares against each symbol's LAST traded price -- not the
    # last price in the panel. Using close.iloc[-1] there both leaks the future
    # and silently NaNs out every name whose tape stops early, which quietly
    # deletes delisted/truncated names from the universe (survivorship bias).
    m = meta.set_index("symbol")
    last_px = close.apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    shares = m["market_cap"] / last_px.reindex(m.index)
    f["mcap"] = (shares * last.reindex(shares.index)).reindex(f.index)
    f["ey"] = f["net_income_ttm"] / f["mcap"]
    f["sales_y"] = f["revenue_ttm"] / f["mcap"]

    f = f.join(m[["sector", "industry", "exchange", "country", "active", "is_etf"]])
    return f


def universe(f, close, asof, start_needed=200):
    asof = pd.Timestamp(asof)
    px = close.loc[:asof]
    full = px.iloc[-start_needed:].notna().all()
    ok = (
        full.reindex(f.index).fillna(False)
        & (f["price"] >= MIN_PRICE)
        & (f["mdvol"] >= MIN_DVOL)
        & f["sector"].notna()
        & f["revenue_ttm"].notna()
        & (f["revenue_ttm"] > 0)
        # NB: deliberately NOT requiring mcap.notna() -- see factor_table.
        & ~(f["mcap"] < 3e8)
        & f["country"].isin(["US"])
    )
    return f[ok].copy()


def zs(s):
    """Cross-sectional rank score in [-0.5, 0.5]; outlier-proof by construction."""
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(pct=True) - 0.5
