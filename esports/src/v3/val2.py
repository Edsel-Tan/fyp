"""v3 step 6b: real-frame check of the detector parser (parse2.py) against the feed.

Per frame and team: detector peaks n vs alive champions a (feed currentHealth > 0). A perfect parser on a fully drawn
minimap has n = a. n > a counts false positives for sure (at least n - a); n < a misses (icons hidden in a pile, off-screen
in a replay, or not drawn). Reported: mean n - a, share n == a, share n > a, and the same split by clock.
Also renders overlays (peaks, identities, wards, dead champions listed) to data/v3/val2/.
usage: python v3/val2.py [N_RENDER_GAMES]
"""
import os, sys, glob, json
import cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minimap as mm
from validate import feed_alive, sec
from parse import meta

D = '../data'


def main():
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    R = []
    files = sorted(glob.glob(f'{D}/v3/parsed2/*.parquet'))
    for k, f in enumerate(files):
        p = pd.read_parquet(f)
        if 'n_b' not in p:
            continue
        gid = str(p.game_id.iloc[0])
        p['sec'] = sec(p.ts)
        p = p.merge(feed_alive(gid), on='sec')
        p = p[(p.clock >= 120) & (p['loc'] >= 0.45)]
        for t, ps, col in ((0, range(1, 6), 'n_b'), (1, range(6, 11), 'n_r')):
            a = sum((p[f'p{i}_currentHealth'] > 0).astype(int) for i in ps)
            R.append(pd.DataFrame(dict(game_id=gid, team=t, clock=p.clock, n=p[col], a=a)))
        if k < nr:
            os.makedirs(f'{D}/v3/val2', exist_ok=True)
            names = meta(gid)
            for _, r in p.iloc[np.linspace(0, len(p) - 1, min(3, len(p))).astype(int)].iterrows():
                im = cv2.imread(f'{D}/v3/crops/{gid}/{r.sec}.png')
                cm = mm.canon(im, (r['loc'], int(r.mm_x), int(r.mm_y), int(r.mm_size)))
                v = cv2.resize(cm, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                for x, y, t, pp in json.loads(r.peaks):
                    cv2.circle(v, (int(x * 768), int(y * 768)), 30, (255, 160, 0) if t == 0 else (0, 0, 255), 2)
                    cv2.putText(v, f'{pp:.2f}', (int(x * 768) - 14, int(y * 768) + 46), 0, 0.45, (255, 255, 255), 1)
                for i in range(1, 11):
                    if np.isfinite(r[f'x{i}']):
                        cv2.putText(v, names[i - 1][:7], (int(r[f'x{i}'] * 768) - 28, int(r[f'y{i}'] * 768) - 34), 0, 0.45, (255, 255, 255), 2)
                for x, y, t in json.loads(r.wards):
                    cv2.rectangle(v, (int(x * 768) - 8, int(y * 768) - 8), (int(x * 768) + 8, int(y * 768) + 8), (0, 255, 0) if t == 0 else (0, 255, 255), 2)
                dead = [names[i - 1][:6] for i in range(1, 11) if r[f'p{i}_currentHealth'] <= 0]
                cv2.putText(v, f'clock {r.clock:.0f}s  alive b {5 - sum(r[f"p{i}_currentHealth"] <= 0 for i in range(1, 6))} r '
                               f'{5 - sum(r[f"p{i}_currentHealth"] <= 0 for i in range(6, 11))}  dead: {",".join(dead)}',
                            (5, 760), 0, 0.5, (255, 255, 255), 2)
                cv2.imwrite(f'{D}/v3/val2/{gid}_{r.sec}.png', v)
    R = pd.concat(R)
    R['e'] = R.n - R.a
    print(f'games {R.game_id.nunique()}, team-frames {len(R)}')
    print(f'mean detected {R.n.mean():.2f} vs alive {R.a.mean():.2f}; n == a {np.mean(R.e == 0):.3f}; n > a {np.mean(R.e > 0):.3f}; '
          f'n < a {np.mean(R.e < 0):.3f}; |n - a| <= 1 {np.mean(R.e.abs() <= 1):.3f}')
    print('by clock bucket (mean n - a, share n == a):')
    b = (R.clock // 600 * 10).clip(upper=40)
    print(R.groupby(b).e.agg(mean='mean', exact=lambda x: np.mean(x == 0), n='size').round(3).to_string())
    print('per-game share n == a, quantiles:', R.groupby('game_id').e.apply(lambda x: np.mean(x == 0)).quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
    R.to_parquet(f'{D}/v3/val2_counts.parquet')


if __name__ == '__main__':
    main()
