"""v5 = v4 (`sel window`: STATS + H, S, G, R, each x (1, m, m^2)) with the union pre-game prior, refit monthly.

Union prior (reddit/eval.py `v4 prior + reddit post-draft`): v2's prior_new feature set plus the r/leagueoflegends Worlds-2026
model's ratings (reddit/build.py: Kalman player ratings, residual lane / damage ratings, Bradley-Terry team rating, draft terms),
one standardised logistic, C = 0.01 chosen on validation. Train rows use the out-of-fold prior (prior_both, as prior2);
test rows use the monthly refit (prior_both_m: refit at each month start on every game finished before it).
The in-game logistic is fit once on the train split, like v4.

Writes data/v5/models.pkl, data/ingame_test_preds_v5.parquet (v4's file + q_v5), and the shuffled-outcome nulls
data/ingame_test_preds_v5_null{k}.parquet (train labels permuted across train games, k = 1..5; the prior is not permuted,
as in eval_ingame's NULL_SEED runs).
Usage: python v5/build.py [nulls]
"""
import os, sys, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v4'))
from eval_pregame import SPLIT, ll, boot_diff
import eval_v4 as E

D = '../data'
PRIOR = 'prior_both_m'


def load():
    f = E.load().merge(pd.read_parquet(f'{D}/reddit/prior.parquet'), on='game_id', how='left')
    assert f[PRIOR].notna().all()
    return f


def main():
    os.makedirs(f'{D}/v5', exist_ok=True)
    M4 = pickle.load(open(f'{D}/v4/models.pkl', 'rb'))['models']['sel window']
    spec = (M4['cols'], 'poly', PRIOR)
    f = load()
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    X = E.X_of(f, None, spec)
    m = E.mk().fit(X[tr], y[tr])
    pv = pd.read_parquet(f'{D}/ingame_test_preds_v4.parquet')
    ft = f[te].reset_index(drop=True)
    assert (pv.game_id.to_numpy() == ft.game_id.to_numpy()).all()
    pv['q_v4'] = pv['q_sel window']
    pv['q_v5'] = m.predict_proba(X[te])[:, 1]
    pv['prior_v5'] = ft[PRIOR].to_numpy()
    pv.to_parquet(f'{D}/ingame_test_preds_v5.parquet')
    pickle.dump(dict(model=m, cols=M4['cols'], basis='poly', prior=PRIOR, xcols=list(X.columns), split=str(SPLIT), built='2026-10-12'),
                open(f'{D}/v5/models.pkl', 'wb'))
    g2m = ft.drop_duplicates('game_id').set_index('game_id').match_id
    pg = lambda c: pd.Series(ll(ft.blue_win.astype(int).to_numpy(), pv[c].to_numpy()), index=ft.game_id).groupby(level=0).mean()
    a, b = pg('q_v5'), pg('q_v4')
    d = boot_diff(a.to_numpy(), b.to_numpy(), g2m.reindex(a.index).to_numpy())
    print(f'test per-game log loss: v5 {a.mean():.4f}  v4 {b.mean():.4f}  v2 {pg("q_lr v2").mean():.4f}  v5 - v4 {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]')

    if len(sys.argv) > 1 and sys.argv[1] == 'nulls':
        gl = f[tr].drop_duplicates('game_id').set_index('game_id').blue_win
        for k in range(1, 6):
            perm = pd.Series(np.random.default_rng(k).permutation(gl.to_numpy()), index=gl.index)
            yk = f.loc[tr, 'game_id'].map(perm).astype(int).to_numpy()
            mk = E.mk().fit(X[tr], yk)
            pn = pv.copy()
            pn['q_v5'] = mk.predict_proba(X[te])[:, 1]
            pn.to_parquet(f'{D}/ingame_test_preds_v5_null{k}.parquet')
            print(f'  null {k}: test per-game log loss {pd.Series(ll(ft.blue_win.astype(int).to_numpy(), pn.q_v5.to_numpy()), index=ft.game_id).groupby(level=0).mean().mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
