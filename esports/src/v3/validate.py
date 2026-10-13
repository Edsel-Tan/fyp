"""v3 step 6: how good is the minimap parser? Checked against the feed, no hand labels needed.

At each parsed crop, the feed frame at the same timestamp says which champions are dead (currentHealth == 0); a dead
champion is not on the minimap. So:
  recall      share of alive champions detected
  false pos.  share of dead champions "detected" (a wrong icon location: overlap, ring, or the respawn timing)
  wards       detected wards per team vs the details feed's wards placed by that team over the last 120 s (a stealth ward
              lives 90-120 s): a correlation across frames, within game (the feed has no ward positions)
Frames before 2:00 are skipped (the feed reports currentHealth 0 for some champions before they first spawn).
Also renders overlays of a few frames per game to data/v3/val/ for eyeballing.
usage: python v3/validate.py [N_RENDER]
"""
import os, sys, glob, json
import cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingame_data import one as frames_one
import minimap as mm

D = '../data'


def sec(ts):
    return ((ts - pd.Timestamp(0, tz='UTC')).dt.total_seconds() // 1).astype('int64')


def feed_alive(gid):
    d = pd.read_parquet(f'{D}/frames60/{gid}.parquet')
    d['sec'] = sec(pd.to_datetime(d.ts, format='ISO8601'))
    o = frames_one(gid)
    o['sec'] = sec(o.ts)
    a = d[['sec'] + [f'p{i}_currentHealth' for i in range(1, 11)]].drop_duplicates('sec')
    return o[['sec', 'clock']].merge(a, on='sec')


def load(gid):
    p = pd.read_parquet(f'{D}/v3/parsed/{gid}.parquet')
    if 'x1' not in p:
        return None
    p['sec'] = sec(p.ts)
    p = p.merge(feed_alive(gid), on='sec', how='inner')
    f = f'{D}/details60/{gid}.parquet'
    if os.path.exists(f):
        w = pd.read_parquet(f)
        w['sec'] = sec(pd.to_datetime(w.ts60, format='ISO8601'))
        w = w.sort_values('sec').drop_duplicates('sec')
        for t, ps in (('b', range(1, 6)), ('r', range(6, 11))):
            cum = sum(w[f'p{i}_wp'].fillna(0) for i in ps)
            w[f'{t}_wp120'] = cum - cum.shift(2).fillna(0)
        p = p.merge(w[['sec', 'b_wp120', 'r_wp120']], on='sec', how='left')
    return p


def stats(p):
    r = []
    for i in range(1, 11):
        alive = p[f'p{i}_currentHealth'] > 0
        det = p[f'x{i}'].notna() & (p['loc'] >= 0.45)
        r.append(pd.DataFrame(dict(game_id=p.game_id, i=i, alive=alive, det=det, x=p[f'x{i}'], y=p[f'y{i}'], clock=p.clock,
                                   s=p[f's{i}'], m=p[f'm{i}'], c=p.get(f'c{i}', np.nan), valid=p['loc'] >= 0.45)))
    return pd.concat(r)


def render(gid, p, n, outdir):
    os.makedirs(outdir, exist_ok=True)
    ch = json.load(open(f'{D}/frames60/{gid}.meta.json'))
    names = [q['championId'] for s in ('blueTeamMetadata', 'redTeamMetadata') for q in s and ch[s]['participantMetadata']]
    q = p[p['loc'] >= 0.45]
    for _, r in q.iloc[np.linspace(0, len(q) - 1, min(n, len(q))).astype(int)].iterrows():
        im = cv2.imread(f'{D}/v3/crops/{gid}/{r.sec}.png')
        cm = mm.canon(im, (r['loc'], *mm.locate(im)[1:]))
        v = cv2.resize(cm, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        for i in range(1, 11):
            dead = r[f'p{i}_currentHealth'] <= 0
            if np.isfinite(r[f'x{i}']):
                c = (255, 160, 0) if i <= 5 else (0, 0, 255)
                cv2.circle(v, (int(r[f'x{i}'] * 768), int(r[f'y{i}'] * 768)), int(r.icon_d * 1.5), c, 2)
                cv2.putText(v, f"{names[i - 1][:6]}{' DEAD' if dead else ''}", (int(r[f'x{i}'] * 768) - 30, int(r[f'y{i}'] * 768) - 34),
                            0, 0.45, (255, 255, 255), 2)
        for x, y, t in json.loads(r.wards):
            cv2.rectangle(v, (int(x * 768) - 9, int(y * 768) - 9), (int(x * 768) + 9, int(y * 768) + 9), (0, 255, 0) if t == 0 else (0, 255, 255), 2)
        miss = [names[i - 1][:6] for i in range(1, 11) if not np.isfinite(r[f'x{i}']) and r[f'p{i}_currentHealth'] > 0]
        cv2.putText(v, f'clock {r.clock:.0f}s missed alive: {",".join(miss)}', (5, 760), 0, 0.5, (255, 255, 255), 2)
        cv2.imwrite(f'{outdir}/{gid}_{r.sec}.png', v)


def main():
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    gids = [os.path.basename(f)[:-8] for f in glob.glob(f'{D}/v3/parsed/*.parquet')]
    S, Wl = [], []
    for g in gids:
        p = load(g)
        if p is None or not len(p):
            continue
        p = p[p.clock >= 120]
        S.append(stats(p))
        if 'b_wp120' in p:
            q = p[p['loc'] >= 0.45].copy()
            q['b_det'] = q.wards.map(lambda s: sum(t == 0 for *_, t in json.loads(s)))
            q['r_det'] = q.wards.map(lambda s: sum(t == 1 for *_, t in json.loads(s)))
            Wl.append(q[['game_id', 'clock', 'b_det', 'r_det', 'b_wp120', 'r_wp120']])
        if nr:
            render(g, p, nr, f'{D}/v3/val')
    S = pd.concat(S)
    v = S[S.valid]
    print(f'games {S.game_id.nunique()}, crops {len(S) // 10}, minimap located on {S.valid.mean():.1%}')
    print(f'recall (alive detected) {v[v.alive].det.mean():.3f}; false pos (dead detected) {v[~v.alive].det.mean():.3f} '
          f'(n dead {int((~v.alive).sum())})')
    W = pd.concat(Wl)
    W = W[W.b_wp120.notna()]
    for t in ('b', 'r'):
        dm = W.groupby('game_id')[[f'{t}_det', f'{t}_wp120']].transform(lambda x: x - x.mean())
        print(f'wards {t}: mean detected {W[f"{t}_det"].mean():.2f}, feed placed/120 s {W[f"{t}_wp120"].mean():.2f}, '
              f'within-game corr {dm.corr().iloc[0, 1]:.3f} (n {len(W)})')
    g = v[v.alive].groupby('game_id').det.mean()
    print('per-game recall quantiles', g.quantile([0, .1, .25, .5, .75, 1]).round(3).to_dict())
    return S


if __name__ == '__main__':
    main()
