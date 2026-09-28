"""Walk-forward in-game win-probability models on the 60 s frames.

Same split as eval_pregame (train 2024-04..2025-06, test 2025-07+).
Models:
  prior         pre-game logistic (elo_team_32, elo_play_32, series_diff, side)
  gold          PLAN's "simple model": logit = a + gd * (b0 + b1 m + b2 m^2), m = minutes / 10
  gold+prior    the same plus prior_logit * (1, m, m^2), so the prior's weight can fade with time
  lr all        every stat and the prior logit, each interacted with (1, m, m^2)
  gbm all       lightgbm on every stat, clock and the prior logit
Scores: per-frame log loss by game-minute bucket, and a per-game mean (each game weighted equally),
with match-clustered bootstrap CIs vs `gold`.
NULL_SEED=k runs the shuffled-outcome null: train labels permuted across train games; outputs get a _null{k} suffix.
"""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eval_pregame import BURN, SPLIT, ll, boot_diff

D = '../data'
NULL = int(os.environ['NULL_SEED']) if os.environ.get('NULL_SEED') else None
STATS = ['gold', 'kills', 'towers', 'inhibs', 'barons', 'drakes', 'elders', 'soul', 'level', 'cs', 'alive',
         'gold_r1', 'gold_r2', 'gold_r3', 'gold_r4', 'gold_r5']
PRE = ['elo_team_32', 'elo_play_32', 'series_diff']


def design(f, cols, poly=True):
    m = f.clock / 600
    X = {}
    for c in cols:
        x = f[c] / (1000 if 'gold' in c else 1) if c in f else f.prior_logit
        X[c] = x
        if poly:
            X[c + '_m'], X[c + '_m2'] = x * m, x * m * m
    return pd.DataFrame(X)


def main():
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')].reset_index(drop=True)
    y = f.blue_win.astype(int).to_numpy()
    tr = (f.t_start < SPLIT).to_numpy()
    te = ~tr
    if NULL is not None:  # shuffled-outcome null: permute train labels across train games, evaluate on true labels
        gl = f[tr].drop_duplicates('game_id').set_index('game_id').blue_win
        perm = pd.Series(np.random.default_rng(NULL).permutation(gl.to_numpy()), index=gl.index)
        f.loc[tr, 'blue_win'] = f.loc[tr, 'game_id'].map(perm).to_numpy()
    yf = f.blue_win.astype(int).to_numpy()  # labels used for fitting (== y unless NULL)

    # prior: fit per game on train games, applied to every frame of a game
    g = f.drop_duplicates('game_id')
    gtr = g[g.t_start < SPLIT]
    pm = LogisticRegression(max_iter=2000).fit(gtr[PRE] / 400, gtr.blue_win.astype(int))  # permuted under NULL
    f['prior_logit'] = pm.decision_function(f[PRE] / 400)

    Q = {'prior': 1 / (1 + np.exp(-f.prior_logit[te].to_numpy()))}
    for name, X in [('gold', design(f, ['gold'])),
                    ('gold+prior', design(f, ['gold', 'prior'])),
                    ('lr all', design(f, STATS + ['prior']))]:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X[tr], yf[tr])
        Q[name] = m.predict_proba(X[te])[:, 1]
    import lightgbm as lgb
    Xg = f[STATS + ['clock']].assign(prior=f.prior_logit)
    gb = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    gb.fit(Xg[tr], yf[tr])
    Q['gbm all'] = gb.predict_proba(Xg[te])[:, 1]

    ft = f[te].copy()
    bucket = (ft.clock // 300 * 5).clip(upper=40).astype(int)
    rows = {}
    for k, q in Q.items():
        l = ll(y[te], q)
        ft['l'] = l
        per_game = ft.groupby('game_id').l.mean()
        rows[k] = {**ft.groupby(bucket).l.mean().rename(lambda b: f'{b}m').to_dict(),
                   'per_game': per_game.mean(), 'per_frame': l.mean(), 'acc': ((q > .5) == y[te]).mean()}
    r = pd.DataFrame(rows).T
    print(f'frames train {tr.sum()} test {te.sum()}; games test {ft.game_id.nunique()}')
    print('frames per bucket:', ft.groupby(bucket).size().to_dict())
    print(r.to_string(float_format=lambda x: f'{x:.4f}'))
    # clustered CI on per-game mean log loss vs gold
    gm = ft[['game_id', 'match_id']].drop_duplicates().set_index('game_id').match_id
    base = pd.Series(ll(y[te], Q['gold']), index=ft.game_id).groupby(level=0).mean()
    for k in Q:
        cur = pd.Series(ll(y[te], Q[k]), index=ft.game_id).groupby(level=0).mean()
        d, lo, hi = boot_diff(cur.to_numpy(), base.reindex(cur.index).to_numpy(), gm.reindex(cur.index).to_numpy())
        print(f'{k:12s} per-game ll - gold: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
    sfx = '' if NULL is None else f'_null{NULL}'
    r.to_csv(f'{D}/../results/ingame_eval{sfx}.csv')
    ft.drop(columns='l').assign(**{f'q_{k}': v for k, v in Q.items()}).to_parquet(f'{D}/ingame_test_preds{sfx}.parquet')


if __name__ == '__main__':
    main()
