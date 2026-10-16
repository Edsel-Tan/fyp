"""MVK vs FUR (Worlds 2026 Play-In, Bo5, 2026-10-15/16; MVK won 3-0, blue side every game): market and model beliefs
across the series. Captured minute bars (pmlive) for the series and game-winner markets; tracker ticks for the in-game
models; draftseries/worlds_priors.py for the draft decomposition. Usage: python draftseries/mvkfur.py"""
import os, sys, glob, sqlite3
import numpy as np, pandas as pd
import pyarrow.dataset as ds
from scipy.optimize import brentq

PM = os.path.expanduser('~/fyp-pmlive/data')
LL = os.path.expanduser('~/fyp-pmlive/lollive/data/paper.sqlite')
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
TOK = {'series': 123209, 'g1': 123215, 'g2': 123221, 'g3': 123219, 'g4': 123227}  # MVK outcome tokens
MATCH = '115660588583040710'
sig = lambda z: 1 / (1 + np.exp(-z))


def bars():
    d = ds.dataset(glob.glob(f'{PM}/pmlive/bars/dt=2026-10-1[56]/*.parquet'), format='parquet')
    b = d.to_table(filter=ds.field('tok').isin(list(TOK.values()))).to_pandas()
    b = b.sort_values(['tok', 'ts']).drop_duplicates(['tok', 'ts'], keep='last')
    b['bb'], b['ba'] = b.bb / 1000, b.ba / 1000
    b['mid'] = np.where((b.bb > 0) & (b.ba < 1) & (b.ba > b.bb), (b.bb + b.ba) / 2, np.nan)
    inv = {v: k for k, v in TOK.items()}
    b['mk'] = b.tok.map(inv)
    m = b.pivot_table(index='ts', columns='mk', values='mid')
    m.index = pd.to_datetime(m.index * 60, unit='s', utc=True).as_unit('ns')
    return m.ffill(limit=10)


def series_p(p, a, b, need=3):
    """P(A wins the series) from score a-b with i.i.d. per-game p."""
    from functools import lru_cache
    @lru_cache(None)
    def f(x, y):
        if x == need: return 1.0
        if y == need: return 0.0
        return p * f(x + 1, y) + (1 - p) * f(x, y + 1)
    return f(a, b)


def implied_p(s, a, b):
    try:
        return brentq(lambda p: series_p(p, a, b) - s, 1e-4, 1 - 1e-4)
    except ValueError:
        return np.nan


def main():
    c = sqlite3.connect(f'file:{LL}?mode=ro', uri=True)
    G = pd.read_sql(f"select * from games where match_id = '{MATCH}' order by number", c)
    G['t0'] = pd.to_datetime(G.first_frame, utc=True, format='ISO8601').dt.as_unit('ns')
    G['t1'] = pd.to_datetime(G.last_frame, utc=True, format='ISO8601').dt.as_unit('ns')
    M = bars()
    W = pd.read_parquet(f'{D}/draftseries/worlds_priors.parquet') if os.path.exists(f'{D}/draftseries/worlds_priors.parquet') else None
    print('== per game: model priors (P(MVK)) and the market')
    score = [(0, 0), (1, 0), (2, 0)]
    rows = []
    for g, (a, b) in zip(G.itertuples(), score):
        k = f'g{g.number}'
        at = lambda t: M[k].asof(t) if M[k].loc[:t].notna().any() else np.nan
        sat = lambda t: M['series'].asof(t)
        prev_end = G.t1[G.number == g.number - 1].iloc[0] if g.number > 1 else None
        r = dict(game=g.number, score=f'{a}-{b}', v2=sig(g.prior_new), v6u=sig(g.v6_prior_u), v6=sig(g.v6_prior), pm_pre=g.v6_pm_pre,
                 mkt_pre=at(g.t0 - pd.Timedelta('15min')), mkt_post=at(g.t0 - pd.Timedelta('1min')), mkt_p1=at(g.t0 + pd.Timedelta('1min')),
                 series=sat(g.t0 - pd.Timedelta('1min')))
        r['series_implied_p'] = implied_p(r['series'], a, b)
        if prev_end is not None:
            r['mkt_after_prev_end+3'] = at(prev_end + pd.Timedelta('3min'))
            r['series_after_prev_end+3'] = sat(prev_end + pd.Timedelta('3min'))
        if W is not None:
            w = W[W.game_id == g.game_id]
            if len(w):
                w = w.iloc[0]
                r.update(z_full=sig(w.z_full), z_pre=sig(w.z_pre), draft_pp=(sig(w.z_full) - sig(w.z_pre)) * 100)
        rows.append(r)
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(R.round(3).to_string(index=False))
    print('\n== series market (P(MVK wins the Bo5)) around each game')
    for g in G.itertuples():
        ts = [g.t0 - pd.Timedelta('10min'), g.t0, g.t0 + pd.Timedelta('10min'), g.t0 + pd.Timedelta('20min'), g.t1, g.t1 + pd.Timedelta('5min')]
        print(f'  g{g.number}: ' + '  '.join(f'{t:%H:%M} {M["series"].asof(t):.3f}' for t in ts))
    print('\n== in-game: P(MVK) by game minute (tracker ticks; blue = MVK)')
    T = pd.read_sql(f"select game_id, t, clock, q_v2, q_v6, q_v6m, blue_mid from ticks where game_id in ({','.join(repr(x) for x in G.game_id)}) order by t", c)
    for g in G.itertuples():
        x = T[T.game_id == g.game_id].copy()
        x['mn'] = x.clock / 60
        out = []
        for mm in (2, 5, 10, 15, 20, 25, 30):
            y = x[x.mn <= mm]
            if len(y) and (x.mn.max() >= mm):
                l = y.iloc[-1]
                out.append(f'{mm:2d}m mid {l.blue_mid:.2f} v6 {l.q_v6:.2f} v6m {l.q_v6m:.2f} v2 {l.q_v2:.2f}')
        print(f'  g{g.number} ({len(x)} ticks, ended {x.mn.max():.0f} min): ' + ' | '.join(out))
    # full-resolution game-winner prices
    R.to_csv(f'{D}/draftseries/mvkfur.csv', index=False)
    M.to_parquet(f'{D}/draftseries/mvkfur_bars.parquet')


if __name__ == '__main__':
    main()
