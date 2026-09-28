"""Latency event study: when does the Polymarket price react to an in-game event,
relative to the event's livestats timestamp?

Events come from full-resolution frames (data/frames10). The feed timestamp tau is the first frame showing the change:
  tower, inhib, baron, drake   a structure/objective counter goes up for one side
  fight                        one side nets >= 3 kills within 20 s (tau = first kill)
For each event with sign s (+1 blue), in 5 s bins over [tau-120, tau+180]:
  move(t)   = s * (VWAP p_blue in the bin - VWAP p_blue in [tau-180, tau-120])
  intensity = trade count in the bin / mean pre-event count
If bettors watch a stream that is ahead of the feed's timestamps, the move starts before 0.
"""
import glob, os
import numpy as np, pandas as pd

D = '../data'
BINS = np.arange(-120, 181, 5)


def events(gid):
    d = pd.read_parquet(f'{D}/frames10/{gid}.parquet')
    d = d[d.state == 'in_game'].copy()
    d['t'] = pd.to_datetime(d.ts, format='ISO8601').astype('datetime64[ms, UTC]').astype('int64') / 1000
    d = d.sort_values('t').drop_duplicates('t')
    ev = []
    for kind, cols in [('tower', 'towers'), ('inhib', 'inhibs'), ('baron', 'barons')]:
        for side, s in (('b', 1), ('r', -1)):
            x = d[f'{side}_{cols}'].diff()
            ev += [(t, s, kind) for t in d.t[x > 0]]
    for side, s in (('b', 1), ('r', -1)):
        n = d[f'{side}_drakes'].str.count(',') + (d[f'{side}_drakes'] != '')
        ev += [(t, s, 'drake') for t in d.t[n.diff() > 0]]
    # fights: net kill swing >= 3 within 20 s, de-duplicated to one event per 60 s
    k = (d.b_kills - d.r_kills).to_numpy(); t = d.t.to_numpy()
    last = -1e18
    for i in range(len(d)):
        j = np.searchsorted(t, t[i] + 20, side='right') - 1
        if i > 0 and abs(k[j] - k[i - 1]) >= 3 and t[i] - last > 60 and k[i] != k[i - 1]:
            ev.append((t[i], int(np.sign(k[j] - k[i - 1])), 'fight')); last = t[i]
    return pd.DataFrame(ev, columns=['tau', 's', 'kind']).assign(game_id=gid)


def main():
    mk = pd.read_parquet(f'{D}/pm_game_markets.parquet').sort_values('vol', ascending=False).drop_duplicates('game_id')
    have = [os.path.basename(p)[:-8] for p in glob.glob(f'{D}/frames10/*.parquet')]
    mk = mk[mk.game_id.isin(have)]
    fills = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event', 'usd'])
    fills = fills[fills.condition_id.isin(mk.condition_id)].merge(mk[['condition_id', 'seq1_blue', 'game_id']], on='condition_id')
    fills['p_blue'] = np.where(fills.seq1_blue == 1, fills.p_event, 1 - fills.p_event)
    rows = []
    for gid, fl in fills.groupby('game_id'):
        ev = events(gid)
        ft, fp, fu = fl.ts.to_numpy().astype(float), fl.p_blue.to_numpy(), fl.usd.to_numpy()
        for e in ev.itertuples():
            base = (ft >= e.tau - 180) & (ft < e.tau - 120)
            if base.sum() < 1:
                continue
            p0 = np.average(fp[base], weights=fu[base])
            rate0 = base.sum() / 12  # trades per 5 s bin before the event
            w = (ft >= e.tau - 120) & (ft < e.tau + 185)
            if not w.any():
                continue
            b = ((ft[w] - e.tau) // 5 * 5).astype(int)
            df = pd.DataFrame(dict(b=b, p=fp[w], u=fu[w]))
            g = df.groupby('b').apply(lambda x: pd.Series(dict(p=np.average(x.p, weights=x.u), n=len(x))), include_groups=False)
            g = g.reindex(BINS)
            g['move'] = e.s * (g.p.ffill() - p0)
            g['intensity'] = g.n.fillna(0) / max(rate0, 1 / 12)
            rows.append(g[['move', 'intensity']].assign(kind=e.kind, game_id=gid, tau=e.tau).rename_axis('bin').reset_index())
    r = pd.concat(rows)
    r.to_parquet(f'{D}/event_study.parquet')
    n = r.drop_duplicates(['game_id', 'tau', 'kind']).kind.value_counts().to_dict()
    print('games', r.game_id.nunique(), 'events', n)
    tab = r.groupby(['kind', 'bin']).move.mean().unstack(0)
    sel = [-60, -30, -15, -10, -5, 0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180]
    print('mean signed price move (p_blue units) by seconds from feed timestamp:')
    print(tab.reindex(sel).to_string(float_format=lambda x: f'{x:+.4f}'))
    print('trade intensity (x pre-event rate):')
    print(r.groupby(['kind', 'bin']).intensity.mean().unstack(0).reindex(sel).to_string(float_format=lambda x: f'{x:.2f}'))


if __name__ == '__main__':
    main()
