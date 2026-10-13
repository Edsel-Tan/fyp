"""v4 step 3: does pre-game group F (pregame_f.py) add to v2's prior (`all new minus h2h` of eval_pregame2)?

Same protocol as eval_pregame2: C chosen on the validation slice (last 91 days of train), refit on train, test scored once;
differences vs v2's prior set with a match-clustered CI. Then, on Polymarket-linked test games only:
  the market's pre-game price (pm_pre) vs prior_new, a fixed 50/50 logit blend, a blend weight fitted by match-grouped CV,
  and the unknown-team guard (pm_pre replaces prior_new when n_min_games < 20).
Writes data/v4/prior4.parquet (prior_f: OOF on train like prior2, train fit on test) and results/v4_pregame_f.csv.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff
from eval_pregame2 import BASE, G

D = '../data'
V2 = BASE + sorted({c for k, v in G.items() if not k.startswith('h2h') for c in v})
F = {
    'series: mp, prev_loser, dec_elo': ['mp', 'prev_loser', 'dec_elo'],
    'same-side form': ['ss_res', 'ss_ewm'],
    'fearless: pool, pool_dep, dep': ['pool', 'pool_dep', 'dep'],
}
ALLF = sorted({c for v in F.values() for c in v})


def lg(q):
    q = np.clip(q, 1e-4, 1 - 1e-4)
    return np.log(q / (1 - q))


def main():
    p = pd.read_parquet(f'{D}/pregame.parquet').merge(pd.read_parquet(f'{D}/pregame2.parquet'), on='game_id') \
         .merge(pd.read_parquet(f'{D}/v4/pregame_f.parquet'), on='game_id')
    p['age_x_elo'] = p.elo_play_32 * np.exp(-p.patch_age / 7)
    p = p[p.t_start >= BURN].reset_index(drop=True)
    tr, te = (p.t_start < SPLIT).to_numpy(), (p.t_start >= SPLIT).to_numpy()
    va = tr & (p.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    fit = tr & ~va
    y = p.blue_win.astype(int).to_numpy()

    def fitpred(cols, C, a, b):
        sc = p.loc[a, cols].std().replace(0, 1)
        m = LogisticRegression(C=C, max_iter=5000).fit(p.loc[a, cols] / sc, y[a])
        return m.predict_proba(p.loc[b, cols] / sc)[:, 1], m, sc

    sets = {'v2 prior (all new minus h2h)': V2, **{f'+ {k}': V2 + v for k, v in F.items()}, '+ all F': V2 + ALLF}
    Q, val = {}, {}
    for k, cols in sets.items():
        val[k] = min((log_loss(y[va], fitpred(cols, C, fit, va)[0]), C) for C in (0.003, 0.01, 0.03, 0.1, 1))
        Q[k] = fitpred(cols, val[k][1], tr, te)[0]
    yt, grp = y[te], p.match_id[te].to_numpy()
    intl = p.league[te].isin(INTL).to_numpy()
    fl2 = (p.fearless[te] == 1).to_numpy() & (p.map_no[te] >= 2).to_numpy()
    base = ll(yt, Q['v2 prior (all new minus h2h)'])
    rows = []
    for k, q in Q.items():
        l = ll(yt, q)
        d, lo, hi = boot_diff(l, base, grp)
        di = boot_diff(l[intl], base[intl], grp[intl])
        dfl = boot_diff(l[fl2], base[fl2], grp[fl2])
        rows.append(dict(model=k, C=val[k][1], val_ll=val[k][0], test_ll=l.mean(), auc=roc_auc_score(yt, q), d=d, lo=lo, hi=hi,
                         d_intl=di[0], intl_ci=f'[{di[1]:+.4f}, {di[2]:+.4f}]', d_fearless_g2plus=dfl[0],
                         fl_ci=f'[{dfl[1]:+.4f}, {dfl[2]:+.4f}]'))
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 300)
    print(f'train {tr.sum()} (val {va.sum()})  test {te.sum()}  intl {intl.sum()}  fearless game 2+ {fl2.sum()}')
    print(R.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    _, m, sc = fitpred(V2 + ALLF, val['+ all F'][1], tr, te)
    print('\n+ all F, standardized coefficients of the F terms:',
          pd.Series(m.coef_[0], index=V2 + ALLF)[ALLF].round(3).to_dict())

    # priors for the in-game model (OOF on train, as eval_pregame2)
    out = p[['game_id']].copy()
    cs, C = V2 + ALLF, val['+ all F'][1]
    z = np.full(len(p), np.nan)
    mids = p.match_id[tr].unique()
    fold = pd.Series(np.random.default_rng(0).integers(0, 5, len(mids)), index=mids)
    ftr = p.match_id.map(fold).to_numpy()
    for k in range(5):
        a, b = tr & (ftr != k), tr & (ftr == k)
        z[b] = lg(fitpred(cs, C, a, b)[0])
    z[te] = lg(Q['+ all F'])
    out['prior_f'] = z
    out.to_parquet(f'{D}/v4/prior4.parquet')

    # the market's pre-game price, on linked test games
    print('\n== market pre-game price (Polymarket per-game markets, test games with pm_pre) ==')
    pt = p[te].assign(q_v2=Q['v2 prior (all new minus h2h)'], q_f=Q['+ all F']).reset_index(drop=True)
    pt = pt[pt.pm_pre.notna() & (pt.pm_pre > 0) & (pt.pm_pre < 1)].reset_index(drop=True)
    yy = pt.blue_win.astype(int).to_numpy()
    zv, zm = lg(pt.q_v2.to_numpy()), lg(pt.pm_pre.to_numpy())
    pt['q_mkt'] = pt.pm_pre
    pt['q_blend50'] = 1 / (1 + np.exp(-(zv + zm) / 2))
    # blend weights by match-grouped 5-fold CV: y ~ a * logit(v2) + b * logit(mkt)
    mids = pt.match_id.unique()
    fold = pt.match_id.map(pd.Series(np.random.default_rng(1).integers(0, 5, len(mids)), index=mids)).to_numpy()
    qcv = np.zeros(len(pt))
    X = np.c_[zv, zm]
    for k in range(5):
        m = LogisticRegression(C=1e6).fit(X[fold != k], yy[fold != k])
        qcv[fold == k] = m.predict_proba(X[fold == k])[:, 1]
    pt['q_blendcv'] = qcv
    wall = LogisticRegression(C=1e6).fit(X, yy)
    low = (pt.n_min_games < 20).to_numpy()
    pt['q_guard'] = np.where(low, pt.pm_pre, pt.q_v2)
    g = pt.match_id.to_numpy()
    b = ll(yy, pt.q_v2.to_numpy())
    print(f'games {len(pt)}, matches {len(mids)}, n_min_games < 20: {low.sum()}; full-sample blend: '
          f'w_v2 {wall.coef_[0][0]:.3f} w_mkt {wall.coef_[0][1]:.3f} intercept {wall.intercept_[0]:+.3f}')
    for c in ('q_v2', 'q_f', 'q_mkt', 'q_blend50', 'q_blendcv', 'q_guard'):
        l = ll(yy, pt[c].to_numpy())
        d, lo, hi = boot_diff(l, b, g)
        dl = boot_diff(l[low], b[low], g[low]) if low.sum() >= 5 else (np.nan,) * 3
        print(f'  {c:10s} ll {l.mean():.4f} auc {roc_auc_score(yy, pt[c]):.3f}  minus v2 prior {d:+.4f} [{lo:+.4f}, {hi:+.4f}]'
              f'   low-history games {dl[0]:+.4f} [{dl[1]:+.4f}, {dl[2]:+.4f}]')
        rows.append(dict(model=f'PM games: {c}', test_ll=l.mean(), d=d, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(f'{D}/../results/v4_pregame_f.csv', index=False)
    pt[['game_id', 'pm_pre', 'q_v2', 'q_f']].to_parquet(f'{D}/v4/pm_pre_test.parquet')


if __name__ == '__main__':
    main()
