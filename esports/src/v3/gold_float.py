"""v3 step 11: gold earned but not held in items ("float"), per player and frame.

The feed has no current-gold field. livestats `window` gives totalGold (earned), and `details` gives each player's item ids
(collected at the 60 s frame timestamps in data/details60). float = totalGold - sum of Data Dragon total costs of the items held, using
the game's patch. It is unspent gold + consumables used (potions, control wards) + sell-back losses + upgrades that cost nothing (support quest, Tear-type transforms).
A team with a large float is either about to buy (pending power spike) or has burnt gold on consumables.
Features (blue - red): fl_team (sum, k gold), fl_r1..fl_r5 (per role, k gold), fl_max (largest single-player float, k gold).
Writes data/v3/gold_float.parquet keyed (game_id, ts).
"""
import os, sys, glob, json
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingame_data import one as frames_one

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
_COST = {}


def costs(patch):
    """{item id: total gold} for the game's patch (falls back to the nearest downloaded version); unknown ids -> latest table."""
    mm = '.'.join(str(patch).split('.')[:2])
    if mm not in _COST:
        f = f'{D}/v3/items/{mm}.1.json'
        if not os.path.exists(f):
            vs = sorted(glob.glob(f'{D}/v3/items/*.json'))
            f = vs[-1]
        c = {int(k): v['gold']['total'] for k, v in json.load(open(f))['data'].items()}
        _COST[mm] = c
    return _COST[mm]


LATEST = None


def value(items, c):
    global LATEST
    if LATEST is None:
        LATEST = costs('16.19')
    s, miss = 0, 0
    for x in str(items or '').split(','):
        if not x:
            continue
        k = int(x)
        v = c.get(k, LATEST.get(k))
        if v is None:
            miss += 1
        else:
            s += v
    return s, miss


def one(args):
    gid, patch = args
    fp = f'{D}/details60/{gid}.parquet'
    if not os.path.exists(fp):
        return None
    o = frames_one(gid)
    if o is None:
        return None
    d = pd.read_parquet(fp)
    d['sec'] = (pd.to_datetime(d.ts60, format='ISO8601') - pd.Timestamp(0, tz='UTC')).dt.total_seconds() // 1
    fr = pd.read_parquet(f'{D}/frames60/{gid}.parquet')
    fr['sec'] = (pd.to_datetime(fr.ts, format='ISO8601') - pd.Timestamp(0, tz='UTC')).dt.total_seconds() // 1
    fr = fr.drop_duplicates('sec').set_index('sec')
    d = d.drop_duplicates('sec').set_index('sec')
    o['sec'] = ((o.ts - pd.Timestamp(0, tz='UTC')).dt.total_seconds() // 1)
    c = costs(patch)
    rows, miss = [], 0
    for s in o.sec:
        if s not in d.index or s not in fr.index:
            rows.append({})
            continue
        fl = []
        for i in range(1, 11):
            v, m = value(d.at[s, f'p{i}_items'], c)
            miss += m
            fl.append(fr.at[s, f'p{i}_totalGold'] - v)
        fl = np.array(fl, float) / 1000
        r = dict(fl_team=fl[:5].sum() - fl[5:].sum(), fl_max=fl[:5].max() - fl[5:].max())
        for k in range(5):
            r[f'fl_r{k + 1}'] = fl[k] - fl[k + 5]
        r['fl_b'], r['fl_r'] = fl[:5].sum(), fl[5:].sum()
        rows.append(r)
    out = pd.concat([o[['game_id', 'ts']].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out['miss_items'] = miss
    return out


def main():
    t = pd.read_parquet(f'{D}/game_table.parquet', columns=['game_id', 'patch'])
    with ProcessPoolExecutor(24) as ex:
        parts = [x for x in ex.map(one, list(zip(t.game_id, t.patch)), chunksize=32) if x is not None]
    f = pd.concat(parts, ignore_index=True)
    f.to_parquet(f'{D}/v3/gold_float.parquet')
    print(f.shape, f.game_id.nunique(), 'frames with float', f.fl_team.notna().sum(), 'games with unknown item ids', (f.groupby('game_id').miss_items.first() > 0).sum())
    print(f[['fl_b', 'fl_r', 'fl_team', 'fl_max']].describe().round(2).T.to_string())


if __name__ == '__main__':
    main()
