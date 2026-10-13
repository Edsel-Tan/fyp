"""When does the Polymarket price react to a feed event, relative to the feed's own timestamp, and when does our live feed deliver it?

Events come from the live tracker's dense saved feeds (lollive data/frames60, ~0.2 s between frames) of the gap panel's tradable games:
a kill burst (kills within 15 s of each other, signed by blue kills minus red kills) or an objective (tower, drake, baron,
inhibitor; signed by the side that took it), timed at the first frame that shows it (feed ts).
Market: every captured fill on the game's two tokens (pmlive trades, 1 s resolution), as a blue-equivalent price (red-token
fills as 1 - price), forward-filled per second. Response(tau) = sign x (price(t_e + tau) - price(t_e - 90 s)). Isolated = no other event in the game in the 90 s before.
Writes results/gap_event_lag.csv; prints the mean response curves and when 10..90 % of the -90..+120 s move has happened.
"""
import os, sys, glob
import numpy as np, pandas as pd, pyarrow.dataset as ds

D = '../data'
LL = os.path.expanduser('~/fyp-pmlive/lollive')
PM = os.path.expanduser('~/fyp-pmlive/data')
TAUS = np.arange(-90, 181, 2)


def events(gid):
    f = f'{LL}/data/frames60/{gid}.parquet'
    if not os.path.exists(f):
        return []
    d = pd.read_parquet(f, columns=['ts', 'b_kills', 'r_kills', 'b_towers', 'r_towers', 'b_drakes', 'r_drakes', 'b_barons', 'r_barons', 'b_inhibs', 'r_inhibs'])
    d['t'] = (pd.to_datetime(d.ts, format='ISO8601', utc=True) - pd.Timestamp(0, tz='UTC')).dt.total_seconds()
    d = d.sort_values('t').drop_duplicates('t')
    for c in d.columns[1:-1]:
        d[c] = pd.to_numeric(d[c], errors='coerce').ffill().fillna(0)
    out = []
    # objectives
    for o in ('towers', 'drakes', 'barons', 'inhibs'):
        for s, c in ((1, f'b_{o}'), (-1, f'r_{o}')):
            v = d[c].to_numpy()
            for i in np.nonzero(np.diff(v) > 0)[0] + 1:
                out.append(dict(game_id=gid, t=d.t.iloc[i], kind=o, sign=s))
    # kill bursts
    dk = d.b_kills.diff().fillna(0).clip(lower=0) - d.r_kills.diff().fillna(0).clip(lower=0)
    tot = d.b_kills.diff().fillna(0).clip(lower=0) + d.r_kills.diff().fillna(0).clip(lower=0)
    idx = np.nonzero(tot.to_numpy() > 0)[0]
    t = d.t.to_numpy()
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and t[idx[j + 1]] - t[idx[j]] <= 15:
            j += 1
        net = dk.iloc[idx[i:j + 1]].sum()
        if net != 0:
            out.append(dict(game_id=gid, t=t[idx[i]], kind='kills', sign=float(np.sign(net)), size=abs(net)))
        i = j + 1
    return out


def main():
    G = pd.read_parquet(f'{D}/gap/games.parquet')
    G = G[G.tradable]
    E = pd.DataFrame([e for g in G.game_id for e in events(g)])
    E = E.merge(G[['game_id', 'tok_b', 'tok_r', 'league']], on='game_id').sort_values(['game_id', 't'])
    E['prev_gap'] = E.groupby('game_id').t.diff()
    print(f'events {len(E)} in {E.game_id.nunique()} games with a dense feed')
    toks = set(E.tok_b) | set(E.tok_r)
    tr = ds.dataset(glob.glob(f'{PM}/pmlive/trades/dt=*/*.parquet'), format='parquet').to_table(
        filter=ds.field('tok').isin(list(toks))).to_pandas()
    tr['p'] = tr.price / 1000
    tb = dict(zip(G.tok_r, G.tok_b))
    tr['blue_p'] = np.where(tr.tok.isin(set(G.tok_b)), tr.p, 1 - tr.p)
    tr['tb'] = tr.tok.map(lambda k: k if k in set(G.tok_b) else tb.get(k))
    tr = tr.sort_values('ts')
    S = {k: (x.ts.to_numpy(), x.blue_p.to_numpy()) for k, x in tr.groupby('tb')}
    rows = []
    for e in E.itertuples():
        if e.tok_b not in S:
            continue
        ts, p = S[e.tok_b]
        def px(t):
            i = np.searchsorted(ts, t, side='right') - 1
            return p[i] if i >= 0 and t - ts[i] < 600 else np.nan
        base = px(e.t - 90)
        r = {f'r{tau}': e.sign * (px(e.t + tau) - base) for tau in TAUS}
        n_after = ((ts > e.t) & (ts <= e.t + 60)).sum()
        rows.append(dict(game_id=e.game_id, league=e.league, kind=e.kind, t=e.t, base=base, n60=n_after, isolated=e.prev_gap > 90, **r))
    R = pd.DataFrame(rows).dropna(subset=['base', 'r120'])
    R = R[(R.base > 0.05) & (R.base < 0.95)]  # the move is bounded near 0/1
    R.to_csv(f'{D}/../results/gap_event_lag.csv', index=False)
    cols = [f'r{t}' for t in TAUS if t <= 120]
    sets = [('kills', R[R.kind == 'kills']), ('kills, isolated', R[(R.kind == 'kills') & R.isolated]),
            ('kills, isolated, EMEA Masters / LCS', R[(R.kind == 'kills') & R.isolated & R.league.isin(['emea_masters', 'lcs'])]),
            ('towers, isolated', R[(R.kind == 'towers') & R.isolated]), ('barons', R[R.kind == 'barons']), ('inhibs', R[R.kind == 'inhibs'])]
    for name, x in sets:
        m = x[cols].mean() * 100
        fin = m['r120']
        q = {f: next((t for t in TAUS if t <= 120 and m[f'r{t}'] >= f * fin), np.nan) for f in (.1, .25, .5, .75, .9)}
        print(f'{name:38s} n {len(x):4d}  move -90..+120 s {fin:+.2f} c; share of it done by t=0: {m["r0"] / fin:.2f};  '
              + '  '.join(f'{int(100 * f)}% by {t:+.0f} s' for f, t in q.items()))
        print('    curve (c): ' + ' '.join(f'{t:+d}:{m[f"r{t}"]:+.1f}' for t in TAUS if t % 10 == 0 and t <= 120))

if __name__ == '__main__':
    main()
