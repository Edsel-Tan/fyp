"""v8 candidate, part 2c: the shape of the Baron / Elder elevation over time, without assuming a linear decay (2026-10-16).

On top of the base in-game model (v5's `sel window` refit, v8/buff_eval.py's cached fit; train frames use the OOF prior,
test frames v7's z7), an offset logistic with holder-signed dummies for the time since each side's latest kill:
    y ~ offset(logit q_base) + sum_bins d_bin,type * (1[blue's latest kill of that type in bin] - 1[red's latest in bin])
bins (s since the kill): 0-60, 60-120, 120-180 (buff live: Baron 180 s, Elder 150 s), 180-240, 240-300, 300-420, 420-600.
Each coefficient is the logit the base model is missing for the holder at that age. Fit on the train frames, scored on test;
also fit on test for the descriptive table. Variant: the 0-180 s Baron bins split by holders alive (5 vs fewer).
Usage: python v8/buff_shape.py   (results/v8_buff_shape.log)
"""
import os, sys, pickle, importlib.util
import numpy as np, pandas as pd
import statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5'), HERE]
from eval_pregame import SPLIT, ll, boot_diff
import eval_v4 as E
import buff_eval as BE

D = os.path.join(SRC, '../data')
EDGES = [0, 60, 120, 180, 240, 300, 420, 600]
lg, sg = BE.lg, BE.sg


def dummies(f):
    X = {}
    for typ in ('bar', 'eld'):
        ab, ar = f[f'age_{typ}_b'].to_numpy(), f[f'age_{typ}_r'].to_numpy()
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            X[f'{typ}_{a}_{b}'] = ((ab >= a) & (ab < b)).astype(float) - ((ar >= a) & (ar < b)).astype(float)
    return pd.DataFrame(X, index=f.index)


def main():
    f = BE.load()
    fits = pickle.load(open(f'{D}/v8/buff_fits.pkl', 'rb'))
    M5 = pickle.load(open(f'{D}/v5/models.pkl', 'rb'))
    base = fits['base (v5 sel window)'][1]
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    pv = pd.read_parquet(f'{D}/ingame_test_preds_v7.parquet')
    f.loc[te, 'z7'] = pv.prior_v7.to_numpy()
    f['pcol'] = np.where(tr, f.prior_both_m, f.z7)
    f['zb'] = lg(base.predict_proba(E.X_of(f, None, (M5['cols'], 'poly', 'pcol')))[:, 1])
    X = dummies(f)
    y = f.blue_win.astype(int).to_numpy()
    grp = pd.factorize(f.match_id)[0]
    print(f'frames: train {tr.sum()}, test {te.sum()}; nonzero dummy frames: ' +
          ', '.join(f'{c} {int((X[c] != 0).sum())}' for c in X.columns))
    res = {}
    for nm, m in (('train', tr), ('test', te)):
        r = sm.GLM(y[m], X[m].to_numpy(), family=sm.families.Binomial(), offset=f.zb.to_numpy()[m]).fit(
            cov_type='cluster', cov_kwds=dict(groups=grp[m]))
        res[nm] = r
    print('\nlogit the base model misses for the holder, by seconds since the kill (cluster se):')
    print(f'  {"term":16s} {"train":>16s} {"test":>16s}   P(holder) at base 0.50 / 0.70 / 0.85 (train coef)')
    for j, c in enumerate(X.columns):
        bt, st = res['train'].params[j], res['train'].bse[j]
        b2, s2 = res['test'].params[j], res['test'].bse[j]
        pp = ' / '.join(f'{sg(lg(q) + bt):.3f}' for q in (0.5, 0.7, 0.85))
        print(f'  {c:16s} {bt:+.3f} ({st:.3f})   {b2:+.3f} ({s2:.3f})   {pp}')
    # out of sample: train coefficients applied to test
    zt = f.zb.to_numpy()[te] + X[te].to_numpy() @ res['train'].params
    ft = f[te]
    a = pd.Series(ll(y[te], sg(zt)), index=ft.game_id).groupby(level=0).mean()
    b = pd.Series(ll(y[te], sg(ft.zb.to_numpy())), index=ft.game_id).groupby(level=0).mean()
    g2m = ft.drop_duplicates('game_id').set_index('game_id').match_id
    r = boot_diff(a.to_numpy(), b.reindex(a.index).to_numpy(), g2m.reindex(a.index).to_numpy())
    on = (X[te].abs().sum(1) > 0).to_numpy()
    r2 = boot_diff(ll(y[te][on], sg(zt[on])), ll(y[te][on], sg(ft.zb.to_numpy()[on])), ft.match_id.to_numpy()[on])
    print(f'\ntest, train-fitted bins on top of the base: per game {r[0]:+.5f} [{r[1]:+.5f}, {r[2]:+.5f}]; frames inside a window '
          f'({on.sum()}) {r2[0]:+.5f} [{r2[1]:+.5f}, {r2[2]:+.5f}]')

    # holders: Baron 0-180 s split by holders alive at the frame (bar_n / bar_t = share of the team still buffed)
    share = np.where(f.bar_t.abs() > 1e-9, (f.bar_n / f.bar_t.where(f.bar_t.abs() > 1e-9)).abs(), np.nan)
    live = ((f.age_bar_b < 180) | (f.age_bar_r < 180)).to_numpy() & np.isfinite(share)
    sgn = np.sign(f.bar_t.to_numpy())
    Xh = X.copy()
    for c in ['bar_0_60', 'bar_60_120', 'bar_120_180']:
        Xh = Xh.drop(columns=c)
    Xh['bar_live_full'] = np.where(live & (share >= 0.99), sgn, 0.0)
    Xh['bar_live_part'] = np.where(live & (share < 0.99) & (share > 0), sgn, 0.0)
    Xh['bar_live_none'] = np.where(live & (share == 0), sgn, 0.0)
    r = sm.GLM(y[tr], Xh[tr].to_numpy(), family=sm.families.Binomial(), offset=f.zb.to_numpy()[tr]).fit(
        cov_type='cluster', cov_kwds=dict(groups=grp[tr]))
    print('\nBaron buff live (0-180 s), by the share of the holders still alive and buffed (train fit):')
    for c in ['bar_live_full', 'bar_live_part', 'bar_live_none']:
        j = list(Xh.columns).index(c)
        print(f'  {c:14s} frames {int((Xh[c] != 0)[tr].sum()):6d}  {r.params[j]:+.3f} ({r.bse[j]:.3f})')


if __name__ == '__main__':
    main()
