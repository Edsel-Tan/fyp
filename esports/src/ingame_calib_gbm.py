"""Next step 2: per-minute calibration of the v2 in-game model, and a tuned GBM against it.

Model `lr v2` = eval_ingame's `lr all` with the out-of-fold prior_new (every stat and the prior, each x (1, m, m^2)).
Calibration: per 5-minute bucket of game clock on the test set, the slope and intercept of y ~ logit(q) (1 = calibrated),
ECE over 10 equal-count bins, and a reliability figure (results/viz/fig_calib_minute.png).
GBM: LightGBM on STATS + clock + prior_new, hyperparameters chosen on the validation slice (last 91 days of train,
fit on the rest, early stopping on validation frames), then refit on all of train with the chosen number of trees.
  gbm tuned        from scratch
  gbm on lr        boosting the residual of `lr v2` (init_score = its logit), a common way to let trees add only
                   interactions the logistic misses
Per-game mean log loss with match-clustered CIs vs `lr v2`.
"""
import itertools
import numpy as np, pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eval_pregame import BURN, SPLIT, ll, boot_diff
from eval_ingame import STATS, design

D = '../data'


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calib(y, q):
    z = logit(q)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
    # intercept at slope 1 (calibration-in-the-large): solve sum(y - sigmoid(z + a)) = 0 by bisection
    lo, hi = -3.0, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (y - 1 / (1 + np.exp(-(z + mid)))).sum() > 0:
            lo = mid
        else:
            hi = mid
    return m.coef_[0, 0], (lo + hi) / 2


def ece(y, q, k=10):
    b = pd.qcut(q, k, labels=False, duplicates='drop')
    d = pd.DataFrame(dict(y=y, q=q, b=b)).groupby('b').agg(y=('y', 'mean'), q=('q', 'mean'), n=('y', 'size'))
    return (d.n * (d.y - d.q).abs()).sum() / d.n.sum()


def main():
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')].reset_index(drop=True)
    f = f.merge(pd.read_parquet(f'{D}/prior2.parquet')[['game_id', 'prior_new']], on='game_id', how='left')
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    va = tr & (f.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    fit = tr & ~va
    Xl = design(f.assign(prior_logit=f.prior_new), STATS + ['prior'])
    lr = lambda a: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Xl[a], y[a])
    m_fit, m_tr = lr(fit), lr(tr)
    z_fit = m_fit.decision_function(Xl)  # lr fit without the validation slice (for the residual GBM's tuning)
    z_tr = m_tr.decision_function(Xl)
    Q = {'lr v2': 1 / (1 + np.exp(-z_tr[te]))}

    Xg = f[STATS + ['clock']].assign(prior=f.prior_new)
    grid = list(itertools.product((4, 7, 15, 63, 255), (100, 1000), (0.5, 0.9)))
    for name, init in (('gbm tuned', None), ('gbm on lr', 'lr')):
        res = []
        for nl, mcs, col in grid:
            g = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.03, num_leaves=nl, min_child_samples=mcs,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=col, reg_lambda=1.0, verbose=-1)
            kw = {} if init is None else dict(init_score=z_fit[fit], eval_init_score=[z_fit[va]])
            g.fit(Xg[fit], y[fit], eval_set=[(Xg[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)], **kw)
            res.append((g.best_score_['valid_0']['binary_logloss'], g.best_iteration_, nl, mcs, col))
            print(f'  {name} leaves {nl:3d} mcs {mcs:4d} col {col}: val ll {res[-1][0]:.4f} at {res[-1][1]} trees', flush=True)
        vll, nt, nl, mcs, col = min(res)
        base_val = ll(y[va], 1 / (1 + np.exp(-z_fit[va]))).mean()
        print(f'{name}: chosen leaves {nl} mcs {mcs} col {col} trees {nt}, val ll {vll:.4f} (lr v2 on val {base_val:.4f})')
        g = lgb.LGBMClassifier(n_estimators=max(nt, 1), learning_rate=0.03, num_leaves=nl, min_child_samples=mcs,
                               subsample=0.8, subsample_freq=1, colsample_bytree=col, reg_lambda=1.0, verbose=-1)
        if init is None:
            g.fit(Xg[tr], y[tr])
            Q[name] = g.predict_proba(Xg[te])[:, 1]
        else:
            g.fit(Xg[tr], y[tr], init_score=z_tr[tr])
            Q[name] = 1 / (1 + np.exp(-(z_tr[te] + g.predict(Xg[te], raw_score=True))))

    ft = f[te].reset_index(drop=True)
    yt = y[te]
    gm = ft.drop_duplicates('game_id').set_index('game_id').match_id
    pg = {k: pd.Series(ll(yt, v), index=ft.game_id).groupby(level=0).mean() for k, v in Q.items()}
    bucket = (ft.clock // 300 * 5).clip(upper=40).astype(int).to_numpy()
    print(f'\ntest games {ft.game_id.nunique()} frames {len(ft)}')
    for k in Q:
        d, lo, hi = boot_diff(pg[k].to_numpy(), pg['lr v2'].reindex(pg[k].index).to_numpy(), gm.reindex(pg[k].index).to_numpy())
        print(f'  {k:10s} per-game ll {pg[k].mean():.4f}  vs lr v2 {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  '
              + ' '.join(f'{b}m:{ll(yt[bucket == b], Q[k][bucket == b]).mean():.3f}' for b in np.unique(bucket)))

    rows = []
    for b in np.unique(bucket):
        m = bucket == b
        for k in Q:
            s, a = calib(yt[m], Q[k][m])
            rows.append(dict(model=k, minute=b, n=m.sum(), games=ft.game_id[m].nunique(), slope=s, intercept=a, ece=ece(yt[m], Q[k][m])))
    C = pd.DataFrame(rows)
    # bootstrap CI (by match) on the lr v2 slope per bucket
    rng = np.random.default_rng(0)
    mids = ft.match_id.unique()
    rows_by_m = ft.groupby('match_id').indices
    bs = {b: [] for b in np.unique(bucket)}
    for _ in range(200):
        ix = np.concatenate([rows_by_m[x] for x in rng.choice(mids, len(mids))])
        for b in bs:
            j = ix[bucket[ix] == b]
            bs[b].append(calib(yt[j], Q['lr v2'][j])[0])
    ci = {b: np.nanpercentile(v, [2.5, 97.5]) for b, v in bs.items()}
    C['slope_lo'] = [ci[b][0] if k == 'lr v2' else np.nan for k, b in zip(C.model, C.minute)]
    C['slope_hi'] = [ci[b][1] if k == 'lr v2' else np.nan for k, b in zip(C.model, C.minute)]
    pd.set_option('display.width', 250)
    print(C.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    C.to_csv(f'{D}/../results/ingame_calib_minute.csv', index=False)
    ft[['game_id', 'match_id', 'clock']].assign(**{f'q_{k}': v for k, v in Q.items()}).to_parquet(f'{D}/ingame_gbm_test_preds.parquet')

    # figure: slope by minute with CI (left), reliability curves for four buckets (right)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    c = C[C.model == 'lr v2']
    ax[0].errorbar(c.minute + 2.5, c.slope, yerr=[c.slope - c.slope_lo, c.slope_hi - c.slope], marker='o', color='#3a6ea5', lw=1.5, capsize=3, label='lr v2')
    for k, col in (('gbm tuned', '#c0504d'), ('gbm on lr', '#9b9b9b')):
        cc = C[C.model == k]
        ax[0].plot(cc.minute + 2.5, cc.slope, marker='.', color=col, lw=1, label=k)
    ax[0].axhline(1, color='k', lw=.8, ls=':')
    ax[0].set(xlabel='game clock (min, 5-min buckets; 40 = 40+)', ylabel='calibration slope (y ~ logit q)', title='Calibration slope by game minute (test)')
    ax[0].legend(frameon=False)
    for b, col in zip((0, 10, 20, 30), ('#bcd0e6', '#7ea3cc', '#3a6ea5', '#1d3b5e')):
        m = bucket == b
        d = pd.DataFrame(dict(y=yt[m], q=Q['lr v2'][m]))
        d['bin'] = pd.qcut(d.q, 10, labels=False, duplicates='drop')
        r = d.groupby('bin').mean()
        ax[1].plot(r.q, r.y, marker='o', ms=3.5, color=col, label=f'{b}-{b + 5} min')
    ax[1].plot([0, 1], [0, 1], color='k', lw=.8, ls=':')
    ax[1].set(xlabel='predicted P(blue)', ylabel='observed blue win rate', title='Reliability, lr v2 (test, decile bins)')
    ax[1].legend(frameon=False)
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(f'{D}/../results/viz/fig_calib_minute.png', dpi=150)


if __name__ == '__main__':
    main()
