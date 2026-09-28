"""Direction 1: do ward (vision) features add to the in-game model?

Livestats has no positions, so map vision is proxied by the `details` feed (collect_details.py), aligned to the
60 s frames: cumulative wards placed / destroyed per player and control wards held (item 2055).
Features (blue minus red): wp, wd, cw, wp_sup, wp_jgl, and 5-frame deltas d5_wp, d5_wd.
Model: `prior_new oof` in-game logistic (eval_ingame2) with and without the vision terms, each x (1, m, m^2),
fit and scored on the games that have details (both arms on identical frames).
Also reports the vision terms' partial effects at 10/20/30 min (does a vision lead help or signal desperation?).
"""
import glob, os
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eval_pregame import BURN, SPLIT, ll, boot_diff
from eval_ingame import STATS, design

D = '../data'
VIS = ['wp', 'wd', 'cw', 'wp_sup', 'wp_jgl', 'd5_wp', 'd5_wd']


def one(path):
    d = pd.read_parquet(path)
    B, R = range(1, 6), range(6, 11)
    o = pd.DataFrame(dict(game_id=d.game_id, ts=pd.to_datetime(d.ts60, format='ISO8601')))
    for c in ('wp', 'wd', 'cw'):
        o[c] = sum(d[f'p{i}_{c}'].fillna(0) for i in B) - sum(d[f'p{i}_{c}'].fillna(0) for i in R)
    o['wp_sup'] = d.p5_wp.fillna(0) - d.p10_wp.fillna(0)
    o['wp_jgl'] = d.p2_wp.fillna(0) - d.p7_wp.fillna(0)
    return o


def main():
    files = glob.glob(f'{D}/details60/*.parquet')
    with ProcessPoolExecutor(16) as ex:
        v = pd.concat(ex.map(one, files, chunksize=64), ignore_index=True)
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')]
    f = f.merge(pd.read_parquet(f'{D}/prior2.parquet'), on='game_id')
    f['ts'] = f.ts.dt.tz_convert('UTC') if f.ts.dt.tz is not None else f.ts.dt.tz_localize('UTC')
    v['ts'] = v.ts.dt.tz_convert('UTC')
    f = f.merge(v, on=['game_id', 'ts'], how='inner').sort_values(['game_id', 'clock']).reset_index(drop=True)
    g = f.groupby('game_id')
    f['d5_wp'] = f.wp - g.wp.shift(5).fillna(0)
    f['d5_wd'] = f.wd - g.wd.shift(5).fillna(0)
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    print(f'games with details: train {f.game_id[tr].nunique()} test {f.game_id[te].nunique()}; frames {len(f)}')
    fp = f.assign(prior_logit=f.prior_new)
    Q, M = {}, {}
    for name, cols in [('prior_new oof', STATS + ['prior']), ('+ vision', STATS + VIS + ['prior'])]:
        X = design(fp, cols)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], y[tr])
        Q[name], M[name] = m.predict_proba(X[te])[:, 1], (m, X.columns)
    ft = f[te]
    gm = ft.drop_duplicates('game_id').set_index('game_id').match_id
    pg = {k: pd.Series(ll(y[te], q), index=ft.game_id).groupby(level=0).mean() for k, q in Q.items()}
    d, lo, hi = boot_diff(pg['+ vision'].to_numpy(), pg['prior_new oof'].reindex(pg['+ vision'].index).to_numpy(),
                          gm.reindex(pg['+ vision'].index).to_numpy())
    bucket = (ft.clock // 600 * 10).clip(upper=30).astype(int).to_numpy()
    for k, q in Q.items():
        print(f'{k:14s} per-game ll {pg[k].mean():.4f}  by 10-min bucket',
              pd.Series(ll(y[te], q)).groupby(bucket).mean().round(4).to_dict())
    print(f'+ vision minus base: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
    # partial effect of +1 SD of each vision term on the logit at 10/20/30 min
    m, cols = M['+ vision']
    lr, sc = m[-1], m[0]
    coef = pd.Series(lr.coef_[0] / sc.scale_, index=cols)
    sd = f.loc[tr, VIS].std()
    out = {c: {t: sd[c] * (coef[c] + coef[c + '_m'] * t / 10 + coef[c + '_m2'] * (t / 10) ** 2) for t in (10, 20, 30)} for c in VIS}
    print('logit change per +1 SD (train) of each vision term, at 10/20/30 min:')
    print(pd.DataFrame(out).T.round(3).to_string())
    pd.DataFrame([dict(model=k, per_game=pg[k].mean()) for k in Q] + [dict(model='diff', per_game=d, lo=lo, hi=hi)]) \
        .to_csv(f'{D}/../results/vision_eval.csv', index=False)


if __name__ == '__main__':
    main()
