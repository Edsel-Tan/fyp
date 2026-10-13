"""Does a roster-adjusted team term beat the persistent team Elo? Protocol of eval_pregame2 (C on the last 91 days of train).

The hierarchical config (Ku, gamma, HL) is chosen on validation log loss of `base + hier`, then every set is scored once on test.
Sets: the base (`all features`) and prior_new (`all new minus h2h`) feature lists, each with
  + res_lin           literal regression residual
  + hier              hier_p, hier_u of the chosen config
  - team elo + hier   elo_team_* removed, hier added (does the roster-aware term replace the stale one?)
Test log loss is also broken down by roster overlap with ~20 games back (ov).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff
from eval_pregame2 import BASE, G
from team_resid import GRID, cname

D = '../data'
TEAM = ['elo_team_16', 'elo_team_32', 'elo_team_64']


def main():
    p = pd.read_parquet(f'{D}/pregame.parquet').merge(pd.read_parquet(f'{D}/pregame2.parquet'), on='game_id') \
        .merge(pd.read_parquet(f'{D}/team_resid.parquet'), on='game_id')
    p['age_x_elo'] = p.elo_play_32 * np.exp(-p.patch_age / 7)
    p = p[p.t_start >= BURN].reset_index(drop=True)
    tr, te = (p.t_start < SPLIT).to_numpy(), (p.t_start >= SPLIT).to_numpy()
    va = tr & (p.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    fit = tr & ~va
    y = p.blue_win.astype(int).to_numpy()
    NEW = BASE + sorted({c for k, v in G.items() if not k.startswith('h2h') for c in v})

    def fitpred(cols, C, a, b):
        sc = p.loc[a, cols].std().replace(0, 1)
        m = LogisticRegression(C=C, max_iter=5000).fit(p.loc[a, cols] / sc, y[a])
        return m.predict_proba(p.loc[b, cols] / sc)[:, 1], m, sc

    def val(cols):
        return min((log_loss(y[va], fitpred(cols, C, fit, va)[0]), C) for C in (0.003, 0.01, 0.03, 0.1, 1))

    # alone, and on top of base: which hierarchical config?
    G_ = []
    for cfg in GRID:
        c = cname(*cfg)
        h = [f'hier_p_{c}', f'hier_u_{c}']
        G_.append(dict(cfg=c, val_alone=val(h)[0], val_base=val(BASE + h)[0], val_replace=val([x for x in BASE if x not in TEAM] + h)[0]))
    G_ = pd.DataFrame(G_).sort_values('val_base')
    print('hierarchical grid (Ku_gamma_HL), validation log loss:')
    print(G_.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    best = G_.cfg.iloc[0]
    H = [f'hier_p_{best}', f'hier_u_{best}']
    print(f'chosen: {best}')

    sets = {}
    for nm, S in (('base', BASE), ('prior_new', NEW)):
        noT = [x for x in S if x not in TEAM]
        sets[nm] = S
        sets[f'{nm} + res_lin'] = S + ['res_lin']
        sets[f'{nm} + hier'] = S + H
        sets[f'{nm} - team elo'] = noT
        sets[f'{nm} - team elo + hier'] = noT + H
    sets['team elo only'] = TEAM
    sets['hier only'] = H
    Q, Cs = {}, {}
    for k, cols in sets.items():
        v, C = val(cols)
        Cs[k] = (v, C)
        Q[k] = fitpred(cols, C, tr, te)[0]
    yt, grp = y[te], p.match_id[te].to_numpy()
    intl = p.league[te].isin(INTL).to_numpy()
    rows = []
    for k, q in Q.items():
        ref = 'prior_new' if k.startswith('prior_new') else 'base'
        l, lr = ll(yt, q), ll(yt, Q[ref])
        d, lo, hi = boot_diff(l, lr, grp)
        di, loi, hii = boot_diff(l[intl], lr[intl], grp[intl])
        rows.append(dict(model=k, ref=ref, C=Cs[k][1], val_ll=Cs[k][0], test_ll=l.mean(), auc=roc_auc_score(yt, q),
                         d=d, lo=lo, hi=hi, intl_ll=l[intl].mean(), d_intl=di, lo_i=loi, hi_i=hii))
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(f'\ntrain {tr.sum()} (val {va.sum()})  test {te.sum()}  intl test {intl.sum()}')
    print(R.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    R.to_csv(f'{D}/../results/team_resid_eval.csv', index=False)

    # by roster continuity
    ov = p.ov[te].to_numpy()
    bk = pd.cut(ov, [-1, 1, 3, 4, 5], labels=['0-1', '2-3', '4', '5'])
    print('\ntest log loss by min roster overlap with ~20 games back (NaN = a side has < 20 games):')
    out = []
    for b in list(bk.categories) + ['NaN']:
        m = (bk == b) if b != 'NaN' else np.isnan(ov)
        r = dict(ov=b, n=int(m.sum()))
        for k in ('base', 'base - team elo + hier', 'prior_new', 'prior_new - team elo + hier'):
            r[k] = ll(yt[m], Q[k][m]).mean()
        for nm in ('base', 'prior_new'):
            d, lo, hi = boot_diff(ll(yt[m], Q[f'{nm} - team elo + hier'][m]), ll(yt[m], Q[nm][m]), grp[m])
            r[f'd_{nm}'] = f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]'
        out.append(r)
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    _, m, sc = fitpred(sets['prior_new + hier'], Cs['prior_new + hier'][1], tr, te)
    print('\nprior_new + hier, standardized coefficients of the strength terms:')
    print(pd.Series(m.coef_[0], index=sets['prior_new + hier']).loc[TEAM + ['elo_play_16', 'elo_play_32', 'elo_play_64'] + H].round(3).to_string())


if __name__ == '__main__':
    main()
