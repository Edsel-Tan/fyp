"""v3 step 8a: v2's logit at every 60 s frame, out-of-fold on train, so a residual model can be fit on top of it.

v2 = eval_ingame2 `prior_new oof` (STATS + prior, each x (1, m, m^2), StandardScaler + LogisticRegression C=1).
Train frames: 5-fold, folds grouped by match. Test frames: v2 fit on all train (reproduces 0.4255).
Writes data/v3/v2_logit.parquet (game_id, ts, clock, split, v2_logit) plus the objective columns.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_pregame import SPLIT, ll
from eval_ingame import STATS, design
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_objectives import load

D = '../data'


def main():
    f = load()
    f['prior_logit'] = f.prior_new
    X = design(f, STATS + ['prior']).to_numpy()
    y = f.blue_win.astype(int).to_numpy()
    tr = (f.t_start < SPLIT).to_numpy()
    lg = np.full(len(f), np.nan)
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    idx = np.flatnonzero(tr)
    for a, b in GroupKFold(5).split(idx, groups=f.match_id.to_numpy()[idx]):
        lg[idx[b]] = mk().fit(X[idx[a]], y[idx[a]]).decision_function(X[idx[b]])
    lg[~tr] = mk().fit(X[tr], y[tr]).decision_function(X[~tr])
    te = ~tr
    pg = pd.Series(ll(y[te], 1 / (1 + np.exp(-lg[te]))), index=f.game_id[te]).groupby(level=0).mean()
    print(f'v2 test per-game log loss {pg.mean():.4f} (eval_ingame2: 0.4255)')
    keep = ['game_id', 'match_id', 'ts', 'clock', 't_start', 'blue_win', 'league', 'drake_up', 'elder_up', 'baron_up',
            't_drake', 't_baron', 'soul_pt', 'soul_up', 'gold', 'alive']
    f[keep].assign(split=np.where(tr, 'train', 'test'), v2_logit=lg).to_parquet(f'{D}/v3/v2_logit.parquet')


if __name__ == '__main__':
    main()
