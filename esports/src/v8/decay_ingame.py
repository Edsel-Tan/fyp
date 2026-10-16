"""v8 candidate, part 1b: time-weighted refits of v7's in-game logistic (2026-10-16).

v7's in-game model is v5's `sel window` logistic, fit once on the train split (frames of games before 2025-07) and frozen.
Here it is refit at each month start from 2025-07 on every frame of every game that ended before the month, with
  frozen        v7 as built (q_v7)
  expanding     equal weights (more data, no decay)
  h {90, 180, 365, 730} d   sample weight 0.5 ** (age / h), age = refit time - game end, normalised to mean 1
The prior column of a training frame is what was causally available for its game: the out-of-fold union prior
(prior_both_m, v5's training prior) before 2025-07, the monthly v7 prior (prior_v7_m, unblended) after. Test frames use
v7's blended prior z7, exactly as q_v7 does. h is chosen on the selection months (2025-07..2026-01), eval months 2026-02+.
Also the live period (v7/live_period_preds: 527 games 2026-09-17..10-11, scored with the latest refit, 2026-09).
Part 2: do the per-role gold coefficients move across patches? Per patch, frames 10-25 min, y ~ offset(logit q_v7) +
gold_r1..r5 (k gold); Cochran's Q and the lag-1 autocorrelation of the per-patch slopes, as in decay_prior.
Writes data/v8/ingame_decay_preds.parquet, results/v8_decay_ingame.log (stdout).
Usage: python v8/decay_ingame.py
"""
import os, sys, pickle, importlib.util
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5')]
from eval_pregame import SPLIT, ll, boot_diff
import eval_v4 as E

D = os.path.join(SRC, '../data')
SEL_END = pd.Timestamp('2026-02-01', tz='UTC')
HS = (90, 180, 365, 730)
sg = lambda z: 1 / (1 + np.exp(-z))
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def load():
    os.chdir(SRC)
    s = importlib.util.spec_from_file_location('v5build', os.path.join(SRC, 'v5', 'build.py'))
    B5 = importlib.util.module_from_spec(s); s.loader.exec_module(B5)
    f = B5.load()
    gt = pd.read_parquet(f'{D}/game_table.parquet', columns=['game_id', 't_end', 'patch'])
    pr = pd.read_parquet(f'{D}/v7/priors.parquet', columns=['game_id', 'prior_v7_m'])
    n = len(f)
    f = f.merge(gt, on='game_id', how='left').merge(pr, on='game_id', how='left')
    assert len(f) == n and f.t_end.notna().all()
    f['ptrain'] = np.where(f.t_start < SPLIT, f.prior_both_m, f.prior_v7_m)
    assert f.ptrain.notna().all(), f.ptrain.isna().sum()
    f['pv'] = f.patch.astype(str).str.split('.').str[:2].str.join('.')
    return f


G = {}  # set before the pool forks: the workers inherit the training matrix copy-on-write


def _fit(args):
    m0, h = args
    a = G['tend'] < m0
    age = (m0 - G['tend'][a]) / np.timedelta64(1, 'D')
    w = np.ones(a.sum()) if h is None else 0.5 ** (age / h)
    m = E.mk()
    m.fit(G['X'][a], G['y'][a], logisticregression__sample_weight=w / w.mean())
    return m0, h, m


def main():
    f = load()
    M5 = pickle.load(open(f'{D}/v5/models.pkl', 'rb'))
    cols = M5['cols']
    y = f.blue_win.astype(int).to_numpy()
    Xtr = E.X_of(f, None, (cols, 'poly', 'ptrain')).to_numpy()
    pv = pd.read_parquet(f'{D}/ingame_test_preds_v7.parquet')
    te = (f.t_start >= SPLIT).to_numpy()
    ft = f[te].reset_index(drop=True)
    assert (ft.game_id.to_numpy() == pv.game_id.to_numpy()).all()
    ft['z7'] = pv.prior_v7.to_numpy()
    ft['q_v7'] = pv.q_v7.to_numpy()
    Xte = E.X_of(ft, None, (cols, 'poly', 'z7')).to_numpy()
    LP = pd.read_parquet(f'{D}/v7/live_period_preds.parquet')
    LP = LP[LP.z6.notna() & LP.blue_win.notna()].reset_index(drop=True)
    Xlp = E.X_of(LP.assign(z7=LP.z6), None, (cols, 'poly', 'z7')).to_numpy()
    print(f'frames {len(f)}; test {te.sum()} ({ft.game_id.nunique()} games); live period {len(LP)} frames ({LP.game_id.nunique()} games)')

    months = pd.date_range(SPLIT, ft.t_start.max(), freq='MS')
    G.update(X=Xtr, y=y, tend=f.t_end.dt.tz_convert('UTC').dt.tz_localize(None).to_numpy())
    jobs = [(m0.tz_localize(None).to_datetime64(), h) for m0 in months for h in (None,) + HS]
    print(f'{len(jobs)} refits ({len(months)} months x {1 + len(HS)} weightings)', flush=True)
    models = {}
    with ProcessPoolExecutor(10) as ex:
        for m0, h, m in ex.map(_fit, jobs):
            models[(pd.Timestamp(m0).tz_localize('UTC'), h)] = m
    mon = ft.t_start.dt.tz_convert('UTC').dt.to_period('M').dt.to_timestamp().dt.tz_localize('UTC')
    arms = {'expanding': None, **{f'h {h} d': h for h in HS}}
    for nm, h in arms.items():
        q = np.full(len(ft), np.nan)
        for m0 in months:
            k = (mon == m0).to_numpy()
            if k.any():
                q[k] = models[(m0, h)].predict_proba(Xte[k])[:, 1]
        ft[f'q[{nm}]'] = q
        LP[f'q[{nm}]'] = models[(months[-1], h)].predict_proba(Xlp)[:, 1]
    # the frozen v5 fit reproduces q_v7 (sanity)
    q0 = M5['model'].predict_proba(Xte)[:, 1]
    print(f'frozen v5 logistic with z7 == q_v7: max |diff| {np.abs(q0 - ft.q_v7).max():.2e}')

    def pg(d, c):
        return pd.Series(ll(d.blue_win.astype(int).to_numpy(), d[c].to_numpy()), index=d.game_id).groupby(level=0).mean()

    def rep(d, label, base='q_v7'):
        g2m = d.drop_duplicates('game_id').set_index('game_id').match_id
        b = pg(d, base)
        print(f'  [{label}] {d.game_id.nunique()} games: {base} {b.mean():.5f}')
        res = {}
        for nm in arms:
            a = pg(d, f'q[{nm}]')
            r = boot_diff(a.to_numpy(), b.reindex(a.index).to_numpy(), g2m.reindex(a.index).to_numpy())
            res[nm] = r[0]
            print(f'    {nm:10s} {a.mean():.5f}  minus frozen {r[0]:+.5f} [{r[1]:+.5f}, {r[2]:+.5f}]')
        return res

    print('\n== 1. monthly refits of the in-game logistic, per-game log loss minus frozen v7 ==')
    S = rep(ft[ft.t_start < SEL_END], 'selection months 2025-07..2026-01')
    rep(ft[ft.t_start >= SEL_END], 'eval months 2026-02..2026-09')
    rep(ft[ft.t_start >= pd.Timestamp('2026-05-01', tz='UTC')], '2026-05..09 (furthest from the frozen fit)')
    rep(ft, 'all test')
    rep(LP.assign(q_v7=LP.q_v7), 'live period 2026-09-17..10-11 (latest refit)')
    choice = min(S, key=S.get)
    print(f'  selection-month choice: {choice}')
    print('\n  by phase, eval months, per-frame LL minus frozen:')
    ev = ft[ft.t_start >= SEL_END]
    ph = pd.cut(ev.clock / 60, [0, 10, 20, 30, 1000], right=False, labels=['0-10', '10-20', '20-30', '30+'])
    for nm in arms:
        d = pd.Series(ll(ev.blue_win.astype(int).to_numpy(), ev[f'q[{nm}]'].to_numpy()) - ll(ev.blue_win.astype(int).to_numpy(), ev.q_v7.to_numpy()))
        print(f'    {nm:10s} ' + '  '.join(f'{k} {v:+.5f}' for k, v in d.groupby(ph.to_numpy(), observed=False).mean().items()))

    # coefficient drift: the gold terms at 15 min across refits (logit per 1k gold)
    print('\n  coefficient drift across monthly refits (expanding vs h 180): logit per 1k at 15 min for gold and per-role gold')
    xc = E.X_of(f.iloc[:5], None, (cols, 'poly', 'ptrain')).columns
    for h in (None, 180):
        s = []
        for m0 in months[::2]:
            m = models[(m0, h)]
            co = pd.Series(m[-1].coef_[0] / m[0].scale_, index=xc)
            v = [co[k] + co[k + '_m'] * 1.5 + co[k + '_m2'] * 2.25 for k in ('gold', 'gold_r1', 'gold_r2', 'gold_r3', 'gold_r4', 'gold_r5')]
            s.append(f'{m0:%y-%m} ' + '/'.join(f'{x:+.3f}' for x in v))
        print(f'    {"expanding" if h is None else f"h {h} d":10s} gold/top/jgl/mid/bot/sup: ' + '  '.join(s))
    co = pd.Series(M5['model'][-1].coef_[0] / M5['model'][0].scale_, index=xc)
    print('    frozen v5 gold/top/jgl/mid/bot/sup: ' + '/'.join(f'{co[k] + co[k + "_m"] * 1.5 + co[k + "_m2"] * 2.25:+.3f}'
                                                         for k in ('gold', 'gold_r1', 'gold_r2', 'gold_r3', 'gold_r4', 'gold_r5')))

    # part 2: per-patch per-role gold slopes on top of q_v7
    print('\n== 2. per-patch role gold slopes on top of q_v7 (test frames 10-25 min, patches with >= 150 games) ==')
    x = ft[(ft.clock >= 600) & (ft.clock < 1500)]
    vc = x.drop_duplicates('game_id').pv.value_counts()
    pats = [p for p in sorted(vc.index, key=lambda s: tuple(int(t) for t in s.split('.'))) if vc[p] >= 150]
    rc = ['gold_r1', 'gold_r2', 'gold_r3', 'gold_r4', 'gold_r5']
    B, Sd = np.full((len(pats), 5), np.nan), np.full((len(pats), 5), np.nan)
    for i, p in enumerate(pats):
        g = x[x.pv == p]
        r = sm.GLM(g.blue_win.astype(int).to_numpy(), sm.add_constant(g[rc] / 1000), family=sm.families.Binomial(),
                   offset=lg(g.q_v7.to_numpy())).fit(cov_type='cluster', cov_kwds=dict(groups=pd.factorize(g.game_id)[0]))
        B[i], Sd[i] = r.params[rc].to_numpy(), r.bse[rc].to_numpy()
    rows = []
    for k, role in enumerate(['top', 'jgl', 'mid', 'bot', 'sup']):
        b, s = B[:, k], Sd[:, k]
        w = 1 / s ** 2
        bb = (w * b).sum() / w.sum()
        q = (w * (b - bb) ** 2).sum()
        tau2 = max(0.0, (q - (len(b) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        rows.append(dict(role=role, mean_slope=bb, patch_sd=b.std(), mean_se=np.sqrt((s ** 2).mean()), Q=q, df=len(b) - 1,
                         p=1 - stats.chi2.cdf(q, len(b) - 1), tau=np.sqrt(tau2), ac1=np.corrcoef(b[:-1], b[1:])[0, 1]))
    print(f'  {len(pats)} patches: ' + ', '.join(pats))
    print('  slope = extra logit per 1k role gold on top of q_v7 (0 = v7 already right on average)')
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    keep = ['game_id', 'match_id', 'ts', 'clock', 't_start', 'blue_win', 'q_v7', 'z7'] + [f'q[{nm}]' for nm in arms]
    ft[keep].to_parquet(f'{D}/v8/ingame_decay_preds.parquet')
    pickle.dump(dict(models={f'{k[0]:%Y-%m}|{k[1]}': v for k, v in models.items()}, cols=cols, choice=choice),
                open(f'{D}/v8/ingame_decay_models.pkl', 'wb'))


if __name__ == '__main__':
    main()
