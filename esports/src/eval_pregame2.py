"""Directions 3-5, pre-game: does each feature group from pregame2 add to the existing `all features` logistic?

Same split as eval_pregame. For each feature set, C is chosen on the validation slice (last 91 days of train,
fit on the rest), then the model is refit on all of train and scored once on test.
Reports test log loss, AUC, the difference vs the base with a match-clustered CI, the international subset,
and the validation log loss the choice was made on. Writes data/pregame2_test_preds.parquet and
data/prior2.parquet (train-and-test out-of-sample-in-time priors for the in-game model).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff

D = '../data'
BASE = ['elo_team_16', 'elo_team_32', 'elo_team_64', 'elo_play_16', 'elo_play_32', 'elo_play_64',
        'form_5', 'form_10', 'form_20', 'h2h', 'exp', 'roster', 'rest_d', 'series_diff']
G = {
    'draft: champ_all': ['champ_all'],
    'draft: champ_patch': ['champ_patch'],
    'draft: champ_all+patch': ['champ_all', 'champ_patch'],
    'draft: +lane matchup': ['champ_all', 'champ_patch', 'lane_mu'],
    'draft: +player-champ': ['champ_all', 'champ_patch', 'lane_mu', 'pc_exp', 'pc_wr'],
    'patch: side_patch + age x elo': ['side_patch', 'age_x_elo'],
    'strength: role elo': [f'role_elo_{k}' for k in range(5)],
    'strength: lane elo': [f'lane_elo_{k}' for k in range(5)],
    'strength: conversion': ['conv'],
    'strength: league elo': ['league_elo'],
    'h2h: team residual': ['h2h_res'],
    'h2h: player pairs': ['pp_res'],
    'h2h: lane pairs': ['lane_h2h'],
}


def main():
    p = pd.read_parquet(f'{D}/pregame.parquet').merge(pd.read_parquet(f'{D}/pregame2.parquet'), on='game_id')
    p['age_x_elo'] = p.elo_play_32 * np.exp(-p.patch_age / 7)
    p = p[p.t_start >= BURN].reset_index(drop=True)
    tr, te = (p.t_start < SPLIT).to_numpy(), (p.t_start >= SPLIT).to_numpy()
    va = tr & (p.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    fit = tr & ~va
    y = p.blue_win.astype(int).to_numpy()
    sets = {'base (all features)': BASE, **{k: BASE + v for k, v in G.items()}}
    sets['all new'] = BASE + sorted({c for v in G.values() for c in v})
    sets['all new minus h2h'] = BASE + sorted({c for k, v in G.items() if not k.startswith('h2h') for c in v})

    def fitpred(cols, C, a, b):
        sc = p.loc[a, cols].std().replace(0, 1)
        m = LogisticRegression(C=C, max_iter=5000).fit(p.loc[a, cols] / sc, y[a])
        return m.predict_proba(p.loc[b, cols] / sc)[:, 1], m, sc

    Q, rows, val = {}, [], {}
    for k, cols in sets.items():
        best = min(((log_loss(y[va], fitpred(cols, C, fit, va)[0]), C) for C in (0.003, 0.01, 0.03, 0.1, 1)))
        val[k] = best
        Q[k] = fitpred(cols, best[1], tr, te)[0]
    yt = y[te]
    grp = p.match_id[te].to_numpy()
    intl = p.league[te].isin(INTL).to_numpy()
    for k, q in Q.items():
        l = ll(yt, q)
        d, lo, hi = boot_diff(l, ll(yt, Q['base (all features)']), grp)
        di, loi, hii = boot_diff(l[intl], ll(yt[intl], Q['base (all features)'][intl]), grp[intl])
        rows.append(dict(model=k, C=val[k][1], val_ll=val[k][0], test_ll=l.mean(), auc=roc_auc_score(yt, q), acc=((q > .5) == yt).mean(),
                         d_vs_base=d, lo=lo, hi=hi, intl_ll=l[intl].mean(), d_intl=di, lo_i=loi, hi_i=hii))
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(f'train {tr.sum()} (val {va.sum()})  test {te.sum()}  intl test {intl.sum()}')
    print(R.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    R.to_csv(f'{D}/../results/pregame2_eval.csv', index=False)

    # standardized coefficients of the full model
    cols = sets['all new']
    _, m, sc = fitpred(cols, val['all new'][1], tr, te)
    print('\nall-new model, standardized coefficients:')
    print(pd.Series(m.coef_[0], index=cols).sort_values(key=abs, ascending=False).round(3).to_string())

    # priors for the in-game model: train rows get 5-fold (by match) out-of-fold predictions so the in-game
    # model does not learn to trust an in-sample prior; test rows use the model fit on all of train
    out = p[['game_id']].copy()
    for name, cs in (('prior_base', BASE), ('prior_new', sets['all new minus h2h'])):
        C = val['all new minus h2h' if name == 'prior_new' else 'base (all features)'][1]
        z = np.full(len(p), np.nan)
        mids = p.match_id[tr].unique()
        fold = pd.Series(np.random.default_rng(0).integers(0, 5, len(mids)), index=mids)
        ftr = p.match_id.map(fold).to_numpy()
        for k in range(5):
            a, b = tr & (ftr != k), tr & (ftr == k)
            q, _, _ = fitpred(cs, C, a, b)
            z[b] = np.log(q / (1 - q))
        q, _, _ = fitpred(cs, C, tr, te)
        z[te] = np.log(q / (1 - q))
        out[name] = z
    out.to_parquet(f'{D}/prior2.parquet')
    p[te][['game_id', 'match_id']].assign(**{f'q_{k}': v for k, v in Q.items()}).to_parquet(f'{D}/pregame2_test_preds.parquet')


if __name__ == '__main__':
    main()
