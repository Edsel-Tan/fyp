"""Walk-forward evaluation of pre-game models.

Train 2024-04-01 .. 2025-06-30 (first three months are rating burn-in), test 2025-07-01 onward.
Everything reports test log loss; differences vs the reference model come with
a match-clustered bootstrap 95 % CI (games in one series are not independent).
"""
import sys
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

D = '../data'
BURN, SPLIT = pd.Timestamp('2024-04-01', tz='UTC'), pd.Timestamp('2025-07-01', tz='UTC')
INTL = {'worlds', 'msi', 'first_stand', 'ewc_lol', 'esports_world_cup'}
rng = np.random.default_rng(0)

p = pd.read_parquet(f'{D}/pregame.parquet')
p = p[p.t_start >= BURN].reset_index(drop=True)
tr, te = p[p.t_start < SPLIT], p[p.t_start >= SPLIT]
ytr, yte = tr.blue_win.astype(int).to_numpy(), te.blue_win.astype(int).to_numpy()
L = pd.read_parquet(f'{D}/pm_links_raw.parquet')
pm_matches = set(L[L.score >= 85].match_id)


def ll(y, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def fit_lr(cols, C=1.0):
    sc = tr[cols].std().replace(0, 1)
    m = LogisticRegression(C=C, max_iter=5000).fit(tr[cols] / sc, ytr)
    return m.predict_proba(te[cols] / sc)[:, 1]


def player_design(d, idx):
    rows, cols, vals = [], [], []
    for i, (bp, rp) in enumerate(zip(d.b_players, d.r_players)):
        for pl in bp:
            if pl in idx: rows.append(i); cols.append(idx[pl]); vals.append(1.0)
        for pl in rp:
            if pl in idx: rows.append(i); cols.append(idx[pl]); vals.append(-1.0)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(d), len(idx)))


def onehot(extra=None):
    idx = {pl: j for j, pl in enumerate(sorted({x for l in tr.b_players for x in l} | {x for l in tr.r_players for x in l}))}
    Xtr, Xte = player_design(tr, idx), player_design(te, idx)
    if extra:
        sc = tr[extra].std()
        Xtr = sp.hstack([Xtr, sp.csr_matrix(tr[extra] / sc)]).tocsr()
        Xte = sp.hstack([Xte, sp.csr_matrix(te[extra] / sc)]).tocsr()
    # choose C on the last 3 months of train, then refit on all of train
    v = (tr.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    best = min((log_loss(ytr[v], LogisticRegression(C=C, max_iter=5000).fit(Xtr[~v], ytr[~v]).predict_proba(Xtr[v])[:, 1]), C)
               for C in (0.01, 0.03, 0.1, 0.3, 1, 3))
    m = LogisticRegression(C=best[1], max_iter=5000).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], best[1], m.predict_proba(Xtr)[:, 1]


def boot_diff(a, b, groups, n=2000):
    """mean(a - b) with a match-clustered bootstrap CI."""
    df = pd.DataFrame(dict(d=a - b, g=groups)).groupby('g').d.agg(['sum', 'size'])
    s, c = df['sum'].to_numpy(), df['size'].to_numpy()
    k = rng.integers(0, len(df), (n, len(df)))
    bs = s[k].sum(1) / c[k].sum(1)
    return s.sum() / c.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    feats = ['elo_team_16', 'elo_team_32', 'elo_team_64', 'elo_play_16', 'elo_play_32', 'elo_play_64',
             'form_5', 'form_10', 'form_20', 'h2h', 'exp', 'roster', 'rest_d', 'series_diff']
    Q = {'0.5': np.full(len(te), .5), 'side only': np.full(len(te), ytr.mean())}
    for f in feats:
        Q[f] = fit_lr([f])
    Q['all elo'] = fit_lr(feats[:6])
    Q['all features'] = fit_lr(feats)
    import lightgbm as lgb
    g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=50, verbose=-1)
    g.fit(tr[feats], ytr)
    Q['gbm all'] = g.predict_proba(te[feats])[:, 1]
    q, C, qtr = onehot()
    Q[f'one-hot players (C={C})'] = q
    print(f'one-hot players: train logloss {log_loss(ytr, qtr):.4f} vs test {log_loss(yte, q):.4f}')
    q, C, _ = onehot(['elo_play_32', 'elo_team_32'])
    Q[f'one-hot + elo (C={C})'] = q

    ref = 'all features'
    grp = te.match_id.to_numpy()
    intl = te.league.isin(INTL).to_numpy()
    pmg = te.match_id.isin(pm_matches).to_numpy()
    out = []
    for k, q in Q.items():
        l = ll(yte, q)
        d, lo, hi = boot_diff(l, ll(yte, Q[ref]), grp)
        out.append(dict(model=k, logloss=l.mean(), brier=((q - yte) ** 2).mean(),
                        auc=roc_auc_score(yte, q) if q.std() > 0 else .5, acc=((q > .5) == yte).mean(),
                        d_vs_ref=d, ci_lo=lo, ci_hi=hi,
                        ll_intl=l[intl].mean(), ll_pm=l[pmg].mean()))
    r = pd.DataFrame(out)
    print(f'train n={len(tr)}  test n={len(te)}  intl n={intl.sum()}  pm-linked n={pmg.sum()}  blue win rate train {ytr.mean():.3f} test {yte.mean():.3f}')
    print(r.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    r.to_csv(f'{D}/../results/pregame_eval.csv', index=False)
    te.assign(**{f'q_{k}': v for k, v in Q.items()})[['game_id', 'match_id'] + [f'q_{k}' for k in Q]].to_parquet(f'{D}/pregame_test_preds.parquet')


if __name__ == '__main__':
    main()
