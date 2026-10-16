"""v8 candidate, part 2b: what a Baron / Elder buff is worth, and whether v7 already prices it (2026-10-16).

Frames: data/ingame.parquet's 60 s frames with data/v8/buffs.parquet (v8/buffs.py). Signals: q_v7 on the test split.
  1 event study around each Baron / Elder kill (holder's view): mean outcome, mean q_v7 and the residual by time since the
    kill; the logit correction v7 needs per bin = an offset GLM  y_h ~ offset(logit q_v7_h) + sum_bin d_bin.
    Windows with another Baron / Elder kill from 4 min before to 10 min after are dropped (no overlapping buffs).
  2 buff terms in the in-game logistic: v5's `sel window` (STATS + H, S, G, R) refit on the train split with prior_both_m
    (exactly v5's fit) plus each candidate set, each x (1, m, m^2); scored on test with v7's blended prior z7 in place of the
    prior, so the base arm is q_v7 itself. The arm is chosen on the validation slice (last 91 train days, fit on the rest).
  3 uncertainty: calibration slope / intercept of each signal and its frame-to-frame logit volatility, by buff window
    (holder's buff live, the 3 min after it expired, no buff).
Writes data/v8/buff_preds.parquet (test frames + each arm's q), results/v8_buff_eval.log (stdout).
Usage: python v8/buff_eval.py
"""
import os, sys, pickle, importlib.util
import numpy as np, pandas as pd
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5')]
from eval_pregame import SPLIT, ll, boot_diff
import eval_v4 as E

D = os.path.join(SRC, '../data')
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))
BF = ['bar_t', 'bar_n', 'bar_on', 'bar_post', 'eld_t', 'eld_n', 'eld_on', 'eld_post']
ARMS = {'base (v5 sel window)': [],
        '+ time left (bar_t, eld_t)': ['bar_t', 'eld_t'],
        '+ holders x time left (bar_n, eld_n)': ['bar_n', 'eld_n'],
        '+ buff on (bar_on, eld_on)': ['bar_on', 'eld_on'],
        '+ holders x time left + post-buff': ['bar_n', 'eld_n', 'bar_post', 'eld_post'],
        '+ time left + holders x time left': ['bar_t', 'eld_t', 'bar_n', 'eld_n']}
RUN1_VAL = {'base (v5 sel window)': 0.41992, '+ time left (bar_t, eld_t)': 0.41995, '+ holders x time left (bar_n, eld_n)': 0.41990,
            '+ buff on (bar_on, eld_on)': 0.41989, '+ holders x time left + post-buff': 0.41993, '+ time left + holders x time left': 0.41999}
BINS = [-240, -180, -120, -60, 0, 60, 120, 180, 240, 300, 360, 480, 600]


def load():
    os.chdir(SRC)
    s = importlib.util.spec_from_file_location('v5build', os.path.join(SRC, 'v5', 'build.py'))
    B5 = importlib.util.module_from_spec(s); s.loader.exec_module(B5)
    f = B5.load()
    n = len(f)
    f = f.merge(pd.read_parquet(f'{D}/v8/buffs.parquet'), on=['game_id', 'ts'], how='left')
    assert len(f) == n and f[BF].notna().all().all()
    return f


def _fit(args):
    name, X, y, tr, va = args
    fit = tr & ~va
    m_va = E.mk().fit(X[fit], y[fit])
    m = E.mk().fit(X[tr], y[tr])
    return name, m_va, m


def event_study(t, ev, qcols):
    """t: test frames; ev: kill events. Holder view per event window.
    After the kill (rel >= 0) the window only uses what is known at the frame: a frame is dropped once any other Baron / Elder
    kill (from 4 min before this kill up to the frame itself) has happened; nothing is dropped because of a later kill.
    Before the kill (rel < 0) the window conditions on a kill that has not happened yet (look-ahead), so those bins only
    describe how the model rates the team that is about to take it."""
    ev = ev[ev.game_id.isin(set(t.game_id))].sort_values(['game_id', 'clock'])
    tg = {k: v for k, v in t.groupby('game_id')}
    eg = {k: v.clock.to_numpy() for k, v in ev.groupby('game_id')}
    out = []
    for typ in ('bar', 'eld'):
        rows = []
        for e in ev[ev.type == typ].itertuples():
            g = tg.get(e.game_id)
            if g is None:
                continue
            oc = eg[e.game_id]
            oc = oc[oc != e.clock]
            c = g.clock.to_numpy()
            rel = c - e.clock
            known = np.array([((oc > e.clock - 240) & (oc <= x)).any() for x in c])
            k = (rel >= BINS[0]) & (rel < BINS[-1]) & ((rel < 0) | ~known)
            if not k.any():
                continue
            h = 1 if e.side == 'blue' else -1
            x = g[k]
            r = pd.DataFrame(dict(game_id=e.game_id, match_id=x.match_id.to_numpy(), rel=rel[k], holders=e.holders,
                                  y=(x.blue_win.to_numpy() if h == 1 else 1 - x.blue_win.to_numpy()).astype(int)))
            for c_ in qcols:
                q = x[c_].to_numpy()
                r[c_] = q if h == 1 else 1 - q
            rows.append(r)
        R = pd.concat(rows, ignore_index=True)
        R['bin'] = pd.cut(R.rel, BINS, right=False, labels=[f'{a:+d}..{b:+d}' for a, b in zip(BINS[:-1], BINS[1:])])
        print(f'\n  {"Baron" if typ == "bar" else "Elder"}: {R.game_id.nunique()} games, {len(R)} frames (rel < 0 bins are look-ahead)')
        print(f'  {"seconds since kill":>18s} {"frames":>6s} {"win (holder)":>12s} ' + ' '.join(f'{c[:14]:>14s}' for c in qcols) +
              '   logit correction each signal needs (cluster se)')
        for b, x in R.groupby('bin', observed=False):
            if len(x) < 30:
                continue
            cs = []
            for c in qcols:
                try:
                    r = sm.GLM(x.y.to_numpy(), np.ones((len(x), 1)), family=sm.families.Binomial(), offset=lg(x[c].to_numpy())).fit(
                        cov_type='cluster', cov_kwds=dict(groups=pd.factorize(x.match_id)[0]))
                    cs.append(f'{r.params[0]:+.3f} ({r.bse[0]:.3f})')
                except Exception:
                    cs.append('   n/a       ')
            print(f'  {b:>18s} {len(x):6d} {x.y.mean():12.3f} ' + ' '.join(f'{x[c].mean():14.3f}' for c in qcols) + '   ' + '  '.join(cs))
        out.append(R.assign(type=typ))
    return pd.concat(out, ignore_index=True)


def by_window(t, cols):
    """Calibration and volatility by buff window (blue view; holder-signed windows are symmetric)."""
    t = t.sort_values(['game_id', 'clock'])
    w = np.where((t.bar_on != 0) | (t.eld_on != 0), 'buff live',
                 np.where((t.bar_post != 0) | (t.eld_post != 0), 'post-buff 0-3 min', 'no buff'))
    late = t.clock.to_numpy() >= 1200
    same = t.game_id.eq(t.game_id.shift()).to_numpy()
    DZ = {c: np.r_[np.nan, np.diff(lg(t[c].to_numpy()))] for c in cols}
    print('\n  window              frames  games | per signal: frame LL, calibration slope (se), intercept, sd of the 60 s logit change')
    for nm in ('buff live', 'post-buff 0-3 min', 'no buff', 'no buff, >= 20 min'):
        m = (w == nm) if nm != 'no buff, >= 20 min' else ((w == 'no buff') & late)
        x = t[m]
        s = f'  {nm:18s} {m.sum():7d} {x.game_id.nunique():6d} |'
        for c in cols:
            z = lg(x[c].to_numpy())
            r = sm.GLM(x.blue_win.astype(int).to_numpy(), sm.add_constant(z), family=sm.families.Binomial()).fit(
                cov_type='cluster', cov_kwds=dict(groups=pd.factorize(x.match_id)[0]))
            vol = np.nanstd(DZ[c][m & same])
            s += f'  {c}: {ll(x.blue_win.astype(int).to_numpy(), x[c].to_numpy()).mean():.4f}, {r.params[1]:.3f} ({r.bse[1]:.3f}), {r.params[0]:+.3f}, {vol:.3f}'
        print(s)


def main():
    f = load()
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    va = tr & (f.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    M5 = pickle.load(open(f'{D}/v5/models.pkl', 'rb'))
    pv = pd.read_parquet(f'{D}/ingame_test_preds_v7.parquet')
    ft = f[te].reset_index(drop=True)
    assert (ft.game_id.to_numpy() == pv.game_id.to_numpy()).all()
    print(f'frames: train {tr.sum()} (validation {va.sum()}), test {te.sum()} ({ft.game_id.nunique()} games)')
    print(f'bar_t == v4 bar_buf: max |diff| {np.abs(f.bar_t - f.bar_buf).max():.2e}; eld_t == eld_buf: {np.abs(f.eld_t - f.eld_buf).max():.2e}')

    jobs = []
    for name, add in ARMS.items():
        X = E.X_of(f, None, (M5['cols'] + add, 'poly', 'prior_both_m'))
        jobs.append((name, X, y, tr, va))
    cache = f'{D}/v8/buff_fits.pkl'
    if os.path.exists(cache):  # the fits take ~25 min and 36 GB; reuse them (delete the file to refit)
        fits = pickle.load(open(cache, 'rb'))
    else:
        with ProcessPoolExecutor(len(jobs)) as ex:
            fits = {n: (mv, m) for n, mv, m in ex.map(_fit, jobs)}
        pickle.dump(fits, open(cache, 'wb'))
    del jobs
    fv = f[va][['game_id']].reset_index(drop=True)
    print('\n== 2. buff terms in the in-game logistic ==')
    print('  validation (fit on train minus the last 91 days), per-game log loss:')
    vres = {}
    for name, add in ARMS.items():
        X = E.X_of(f[va].reset_index(drop=True), None, (M5['cols'] + add, 'poly', 'prior_both_m'))
        if fits[name][0] is None:  # cached from the first run (results/v8_buff_eval_run1.log has the validation table)
            vres = None
            break
        q = fits[name][0].predict_proba(X)[:, 1]
        vres[name] = pd.Series(ll(y[va], q), index=fv.game_id).groupby(level=0).mean().mean()
    if vres is None:
        vres = dict(RUN1_VAL)
    for k, v in vres.items():
        print(f'    {k:40s} {v:.5f}  ({v - vres["base (v5 sel window)"]:+.5f})')
    choice = min(vres, key=vres.get)
    print(f'  validation choice: {choice}')

    ft = ft.assign(prior_v7=pv.prior_v7.to_numpy(), q_v7=pv.q_v7.to_numpy())
    for name, add in ARMS.items():
        X = E.X_of(ft, None, (M5['cols'] + add, 'poly', 'prior_v7'))
        ft[f'q[{name}]'] = fits[name][1].predict_proba(X)[:, 1]
    dmax = np.abs(ft['q[base (v5 sel window)]'] - ft.q_v7).max()
    print(f'  base arm with z7 == q_v7: max |diff| {dmax:.2e}')
    g2m = ft.drop_duplicates('game_id').set_index('game_id').match_id
    pg = lambda d, c: pd.Series(ll(d.blue_win.astype(int).to_numpy(), d[c].to_numpy()), index=d.game_id).groupby(level=0).mean()
    buff = (ft.bar_on != 0) | (ft.eld_on != 0) | (ft.bar_post != 0) | (ft.eld_post != 0)
    BASE = 'q[base (v5 sel window)]'
    print('  test split, log loss minus the refit base arm (match-clustered CI): per game | per frame on buff / post-buff frames | per frame >= 30 min')
    for name in ARMS:
        c = f'q[{name}]'
        a, b = pg(ft, c), pg(ft, BASE)
        r = boot_diff(a.to_numpy(), b.reindex(a.index).to_numpy(), g2m.reindex(a.index).to_numpy())
        fb = ft[buff]
        rb = boot_diff(ll(fb.blue_win.astype(int).to_numpy(), fb[c].to_numpy()), ll(fb.blue_win.astype(int).to_numpy(), fb[BASE].to_numpy()), fb.match_id.to_numpy())
        fl = ft[ft.clock >= 1800]
        rl = boot_diff(ll(fl.blue_win.astype(int).to_numpy(), fl[c].to_numpy()), ll(fl.blue_win.astype(int).to_numpy(), fl[BASE].to_numpy()), fl.match_id.to_numpy())
        print(f'    {name:40s} {a.mean():.5f} {r[0]:+.5f} [{r[1]:+.5f}, {r[2]:+.5f}] | buff ({buff.sum()} fr) {rb[0]:+.5f} [{rb[1]:+.5f}, {rb[2]:+.5f}]'
              f' | 30+ {rl[0]:+.5f} [{rl[1]:+.5f}, {rl[2]:+.5f}]')

    # coefficient read-out: logit elevation of a full buff (all 5 holders, the full duration left) and at the frame after the kill
    print('\n  partial effects (logit, holder minus no buff, per unit of the term) at 20 / 25 / 30 / 35 / 40 min:')
    for name, add in ARMS.items():
        if not add:
            continue
        mdl = fits[name][1]
        Xc = E.X_of(f.iloc[:5], None, (M5['cols'] + add, 'poly', 'prior_both_m')).columns
        co = pd.Series(mdl[-1].coef_[0] / mdl[0].scale_, index=Xc)
        s = f'    {name:40s}'
        for k in add:
            eff = [co[k] + co[k + '_m'] * mm + co[k + '_m2'] * mm * mm for mm in (2, 2.5, 3, 3.5, 4)]
            s += f'  {k}: ' + '/'.join(f'{v:+.2f}' for v in eff)
        print(s)
    for name, add in ARMS.items():  # the cumulative counts in every arm
        mdl = fits[name][1]
        Xc = E.X_of(f.iloc[:5], None, (M5['cols'] + add, 'poly', 'prior_both_m')).columns
        co = pd.Series(mdl[-1].coef_[0] / mdl[0].scale_, index=Xc)
        print(f'    {name:40s} cumulative counts  ' + '  '.join(f'{k}: ' + '/'.join(f'{co[k] + co[k + "_m"] * mm + co[k + "_m2"] * mm * mm:+.2f}' for mm in (2, 2.5, 3, 3.5, 4))
                                                    for k in ('barons', 'elders')))
    fits_buff = {}
    for name, add in ARMS.items():
        mdl = fits[name][1]
        Xc = E.X_of(f.iloc[:5], None, (M5['cols'] + add, 'poly', 'prior_both_m')).columns
        fits_buff[name] = pd.Series(mdl[-1].coef_[0] / mdl[0].scale_, index=Xc)
    pickle.dump(dict(fits={k: v[1] for k, v in fits.items()}, arms=ARMS, choice=choice, coefs=fits_buff), open(f'{D}/v8/buff_models.pkl', 'wb'))

    print('\n== 1. event study around kills (test split, holder view): outcome vs q_v7 and the validation-chosen arm ==')
    ev = pd.read_parquet(f'{D}/v8/buff_events.parquet')
    qc = ['q[base (v5 sel window)]', f'q[{choice}]', 'q[+ holders x time left + post-buff]']
    ES = event_study(ft, ev, qc)
    print('\n  by holders at the kill (Baron, 0..+180 s frames): win rate, q_v7, residual')
    x = ES[(ES.type == 'bar') & (ES.rel >= 0) & (ES.rel < 180)]
    print(x.groupby(x.holders.clip(lower=2)).agg(frames=('y', 'size'), win=('y', 'mean'), q_base=(qc[0], 'mean'), q_n=(qc[2], 'mean')).assign(
        resid_base=lambda d: d.win - d.q_base, resid_n=lambda d: d.win - d.q_n).round(3).to_string())

    print('\n== 3. uncertainty by buff window (test split) ==')
    by_window(ft, qc)
    keep = ['game_id', 'match_id', 'ts', 'clock', 'blue_win', 't_start', 'league'] + BF + [c for c in ft.columns if c.startswith('q[')] + ['q_v7', 'prior_v7']
    ft[keep].to_parquet(f'{D}/v8/buff_preds.parquet')
    ES.to_parquet(f'{D}/v8/buff_event_study.parquet')


if __name__ == '__main__':
    main()
