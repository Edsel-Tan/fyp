"""v3 step 12: does gold float (gold_float.py) add to v2? Full 60 s frame set, same split and model family as eval_ingame2.

Glitch frames (an empty item list late in the game: team float above 2x the 99th percentile of its 5-minute bucket, 0.3 % of
frames) get float 0 and an indicator. Features (blue - red, k gold): fl_team, fl_max, fl_r1..fl_r5, fl_d1 (change since the
previous frame: gold just earned and not yet spent), each x (1, m, m^2). Also reports the coefficient path and the gain by game minute.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pregame import SPLIT, ll, boot_diff
from eval_ingame import STATS, design
from eval_ingame2 import fitq
from eval_objectives import load

D = '../data'
FL = ['fl_team', 'fl_max', 'fl_r1', 'fl_r2', 'fl_r3', 'fl_r4', 'fl_r5', 'fl_d1', 'fl_bad']


def add_float(f):
    g = pd.read_parquet(f'{D}/v3/gold_float.parquet')
    f = f.merge(g.drop(columns=['miss_items']), on=['game_id', 'ts'], how='left')
    b = (f.clock // 300).clip(upper=9)
    lim = 2 * pd.concat([f.fl_b, f.fl_r]).groupby(pd.concat([b, b])).quantile(0.99)
    bad = (f.fl_b > b.map(lim)) | (f.fl_r > b.map(lim)) | f.fl_team.isna()
    f['fl_bad'] = bad.astype(float)
    cols = ['fl_team', 'fl_max'] + [f'fl_r{k}' for k in range(1, 6)]
    f.loc[bad, cols] = 0
    f = f.sort_values(['game_id', 'clock'])
    f['fl_d1'] = (f.fl_team - f.groupby('game_id').fl_team.shift(1)).fillna(0).where(~bad, 0)
    return f.sort_index()


def main():
    f = add_float(load())
    f['prior_logit'] = f.prior_new
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    print(f'frames {len(f)}; float available {(f.fl_bad == 0).mean():.1%}')
    Q = {'v2': fitq(design(f, STATS + ['prior']), y, tr, te),
         'v2 + float': fitq(design(f, STATS + FL + ['prior']), y, tr, te),
         'v2 + fl_team, fl_d1': fitq(design(f, STATS + ['fl_team', 'fl_d1', 'fl_bad'] + ['prior']), y, tr, te)}
    ft = f[te].reset_index(drop=True)
    yt = y[te]
    gm = ft.drop_duplicates('game_id').set_index('game_id').match_id
    pg = {k: pd.Series(ll(yt, v), index=ft.game_id).groupby(level=0).mean() for k, v in Q.items()}
    rows = []
    bucket = (ft.clock // 600 * 10).clip(upper=30).astype(int).to_numpy()
    for k in Q:
        d, lo, hi = boot_diff(pg[k].to_numpy(), pg['v2'].to_numpy(), gm.reindex(pg['v2'].index).to_numpy())
        byb = pd.Series(ll(yt, Q[k]) - ll(yt, Q['v2'])).groupby(bucket).mean().round(4).to_dict()
        print(f'{k:22s} {pg[k].mean():.4f}  vs v2 {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  by 10-min bucket {byb}')
        rows.append(dict(model=k, per_game=pg[k].mean(), d=d, lo=lo, hi=hi))
    # sign of the effect: logit change per +1k float lead at 10/20/30 min (single-feature fit for readability)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = design(f, STATS + ['fl_team', 'fl_d1', 'fl_bad'] + ['prior'])
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], y[tr])
    c = pd.Series(m[-1].coef_[0] / m[0].scale_, index=X.columns)
    for k in ('fl_team', 'fl_d1'):
        print(f'  {k}: logit per +1k at 10/20/30 min', [round(c[k] + c[k + '_m'] * t + c[k + '_m2'] * t * t, 3) for t in (1, 2, 3)],
              ' (gold lead per +1k:', [round((c['gold'] + c['gold_m'] * t + c['gold_m2'] * t * t), 3) for t in (1, 2, 3)], ')')
    pd.DataFrame(rows).to_csv(f'{D}/../results/v3_float.csv', index=False)


if __name__ == '__main__':
    main()
