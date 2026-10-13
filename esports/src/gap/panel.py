"""Frame-level panel for the market-vs-model gap study: every 60 s frame of the live-period games (v5/live_period.py) that links to a
captured Polymarket per-game market (v5/book_sim.py's linker), with
  - the model forecasts (q_v2 / q_v4 / q_v5, priors) and the frame's features
  - the captured blue-token mid as of several wall-clock offsets k (minutes) from the frame: mid_k = the book at the end of
    minute floor(ts / 60) + k, as-of (the last bar at or before it; bars are written only for minutes with an update).
    k = -1: the market as of (ts - 60, ts], strictly older than the state the model sees. k = 0: up to 60 s newer.
  - the market's pre-game price (median two-sided mid over the 10 minutes before the game's first frame)
  - tradable: the game had a working live feed (book_sim's rule)
Writes data/gap/panel.parquet and data/gap/games.parquet.
"""
import os, sys, sqlite3
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v5'))
import book_sim as B

D = '../data'
KS = list(range(-5, 4))


def asof_mid(x, minute):
    """Two-sided mid of the last bar at or before `minute` (bars indexed by integer minute), with its age in minutes."""
    i = x.index.searchsorted(minute, side='right') - 1
    while i >= 0:
        r = x.iloc[i]
        if r.bb > 0 and 0 < r.ba < 1 and r.ba > r.bb:
            return (r.bb + r.ba) / 2, minute - x.index[i], r.ba - r.bb
        i -= 1
        if minute - x.index[i] > 30:
            break
    return np.nan, np.nan, np.nan


def main():
    os.makedirs(f'{D}/gap', exist_ok=True)
    F = pd.read_parquet(f'{D}/v5/live_period_preds.parquet')
    G = pd.read_parquet(f'{D}/v5/live_period_games.parquet')
    M = B.markets()
    Lk = B.link(G[G.game_id.isin(F.game_id)], M, B.team_names())
    bars = B.load_bars(set(Lk.tok_b) | set(Lk.tok_r))
    end = (F.groupby('game_id').ts.max() - pd.Timestamp(0, tz='UTC')).dt.total_seconds() // 60

    def endp(r):
        x = bars.get(r.tok_b)
        if x is None:
            return np.nan
        w = x.loc[int(end[r.game_id]):int(end[r.game_id]) + 120]
        return float(((w.bb + w.ba) / 2).where(w.ba > 0, w.bb).median()) if len(w) else np.nan
    Lk['last_blue'] = Lk.apply(endp, axis=1)
    Lk = Lk.merge(G[['game_id', 'blue_win']], on='game_id')
    res = Lk.last_blue.notna() & ((Lk.last_blue > 0.85) | (Lk.last_blue < 0.15))
    bad = res & ((Lk.last_blue > 0.5) != (Lk.blue_win == 1))
    Lk = Lk[~bad & Lk.tok_b.isin(bars)].drop(columns='blue_win')

    lol = sqlite3.connect(f'file:{B.LL}/data/paper.sqlite?mode=ro', uri=True)
    lg = pd.read_sql('select game_id, status, joined_at from games', lol)
    t_live = pd.to_datetime(lg.joined_at.min(), unit='s', utc=True)
    fin = set(lg.game_id[lg.status == 'finished'])
    gs = G.set_index('game_id')
    Lk['tradable'] = [(gs.at[g, 't_start'] < t_live and gs.at[g, 'league'] != 'wsci') or g in fin for g in Lk.game_id]

    F = F.merge(Lk[['game_id', 'tok_b', 'slug', 'tradable']], on='game_id').sort_values(['game_id', 'ts']).reset_index(drop=True)
    m0 = ((F.ts - pd.Timestamp(0, tz='UTC')).dt.total_seconds() // 60).astype(int).to_numpy()
    for k in KS:
        out = [asof_mid(bars[t], m + k) for t, m in zip(F.tok_b, m0)]
        F[f'mid_{k}'] = [o[0] for o in out]
        F[f'age_{k}'] = [o[1] for o in out]
        F[f'spr_{k}'] = [o[2] for o in out]
    # pre-game price: median two-sided mid over the 10 bars before the first frame's minute
    first = F.groupby('game_id').ts.min()
    pre = {}
    for g, t in first.items():
        x = bars[F.tok_b[F.game_id == g].iloc[0]]
        mm = int((t - pd.Timestamp(0, tz='UTC')).total_seconds() // 60)
        w = x.loc[mm - 10:mm - 1]
        w = w[(w.bb > 0) & (w.ba < 1) & (w.ba > w.bb)]
        pre[g] = float(((w.bb + w.ba) / 2).median()) if len(w) else np.nan
    F['pm_pre'] = F.game_id.map(pre)
    g = G[G.game_id.isin(F.game_id)].merge(Lk, on='game_id')
    g['pm_pre'] = g.game_id.map(pre)
    F.to_parquet(f'{D}/gap/panel.parquet')
    g.to_parquet(f'{D}/gap/games.parquet')
    print(f'panel: {len(F)} frames, {F.game_id.nunique()} games ({F[F.tradable].game_id.nunique()} tradable); '
          f'pre-game price for {np.isfinite(list(pre.values())).sum()} games; mid_-1 present {F["mid_-1"].notna().mean():.1%}, '
          f'mid_0 {F.mid_0.notna().mean():.1%}; median age of mid_-1 {F["age_-1"].median()} min')


if __name__ == '__main__':
    main()
