"""Per-frame in-game dataset from the 60 s frames: one row per (game, frame).

Game clock: frames before gold appears (loading) are dropped. The first frame with gold
anchors the clock: all ten players start with 500 (5,000 total), and passive income
starts at 1:05 at ~20.4 gold/s in total, so gt = 65 + (gold - 5000) / 20.4 while gold < 5,400
(otherwise 90 s is assumed). After that, the clock advances with wall time minus time in
`paused` frames. The error is roughly +-30 s at 60 s sampling. All features are blue minus red.
The label is the game's `blue_win`; the pre-game prior comes from pregame.parquet.
"""
import glob, os
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

D = '../data'
B, R = range(1, 6), range(6, 11)


def one(gid):
    d = pd.read_parquet(f'{D}/frames60/{gid}.parquet')
    if any(f'p{i}_currentHealth' not in d for i in range(1, 11)):
        return None
    d = d[d.state.isin(['in_game', 'paused', 'finished'])]
    d = d.assign(_t=pd.to_datetime(d.ts, format='ISO8601')).sort_values('_t').drop_duplicates('_t')
    d = d[(d.b_gold + d.r_gold) > 0].reset_index(drop=True)
    if len(d) < 3:
        return None
    ts = d._t
    g0 = d.b_gold.iloc[0] + d.r_gold.iloc[0]
    gt0 = 65 + max(0, g0 - 5000) / 20.4 if g0 < 5400 else 90
    dt = ts.diff().dt.total_seconds().fillna(0).to_numpy()
    clock = gt0 + np.cumsum(np.where(d.state.to_numpy() == 'paused', 0, dt))
    o = pd.DataFrame(dict(game_id=gid, ts=ts, clock=clock, state=d.state))
    o['gold'] = d.b_gold - d.r_gold
    o['kills'] = d.b_kills - d.r_kills
    o['towers'] = d.b_towers - d.r_towers
    o['inhibs'] = d.b_inhibs - d.r_inhibs
    o['barons'] = d.b_barons - d.r_barons
    nb = d.b_drakes.str.split(',').map(lambda l: [x for x in l if x])
    nr = d.r_drakes.str.split(',').map(lambda l: [x for x in l if x])
    o['drakes'] = nb.map(lambda l: sum(x != 'elder' for x in l)) - nr.map(lambda l: sum(x != 'elder' for x in l))
    o['elders'] = nb.map(lambda l: l.count('elder')) - nr.map(lambda l: l.count('elder'))
    o['soul'] = (nb.map(lambda l: sum(x != 'elder' for x in l)) >= 4).astype(int) - \
        (nr.map(lambda l: sum(x != 'elder' for x in l)) >= 4).astype(int)
    o['level'] = sum(d[f'p{i}_level'] for i in B) - sum(d[f'p{i}_level'] for i in R)
    o['cs'] = sum(d[f'p{i}_creepScore'] for i in B) - sum(d[f'p{i}_creepScore'] for i in R)
    o['alive'] = sum((d[f'p{i}_currentHealth'] > 0).astype(int) for i in B) - \
        sum((d[f'p{i}_currentHealth'] > 0).astype(int) for i in R)
    o['tot_gold'] = d.b_gold + d.r_gold
    for i in range(1, 11):  # per-role gold, blue minus red counterpart
        if i <= 5:
            o[f'gold_r{i}'] = d[f'p{i}_totalGold'] - d[f'p{i + 5}_totalGold']
    return o


def main():
    t = pd.read_parquet(f'{D}/pregame.parquet')
    gids = [g for g in t.game_id if os.path.exists(f'{D}/frames60/{g}.parquet')]
    with ProcessPoolExecutor(16) as ex:
        parts = [x for x in ex.map(one, gids, chunksize=64) if x is not None]
    f = pd.concat(parts, ignore_index=True)
    keep = ['game_id', 'match_id', 'league', 't_start', 'blue_win', 'y_src', 'elo_team_32', 'elo_play_32', 'series_diff']
    f = f.merge(t[keep], on='game_id')
    f.to_parquet(f'{D}/ingame.parquet')
    print(f.shape, f.game_id.nunique())
    print(f.groupby((f.clock // 300 * 5).clip(upper=45)).gold.agg(['size', 'mean', 'std']).to_string())
    first = f.groupby('game_id').first()
    print('first frame clock/tot_gold:', first.clock.describe().round(1).to_dict(), first.tot_gold.describe().round(0).to_dict())


if __name__ == '__main__':
    main()
