"""v8 candidate = v7 + the changes that survived v8's tests (2026-10-16). Each was chosen on validation / selection months:
  pre-game  (a) champion tables' half-life 30 d instead of 120 d (decay_inputs*.py: champ_all, champ_patch, lane_mu, pc_exp,
                pc_wr, side_patch, conv)
            (b) the union logistic refit weekly instead of monthly, equal weights (decay_prior.py: every decay h <= 180 d hurt)
            (c) the market blend (w0 + w1 cross) refit on the new prior, selection months
  in-game   (d) the `sel window` logistic refit at each month start on every finished game (expanding, equal weights) instead
                of frozen at 2025-06 (decay_ingame.py: decay hurt, h 90 d chosen on selection was the worst on eval)
            (e) + the buff terms bar_on / eld_on (buff_eval.py's validation choice; the holders x time-left form is reported too)
Stages are cumulative and each is scored vs q_v7 on the test split, the eval months and the live period:
  v8_pre    frozen in-game model, z8 in place of z7                       (a, b, c)
  v8_refit  monthly in-game refit, z8                                     (+ d)
  v8        monthly in-game refit + bar_on / eld_on, z8                   (+ e)
  v8_n      monthly in-game refit + bar_n / eld_n, z8                     (alternative to e)
Training prior of an in-game frame: OOF union prior before 2025-07, the unblended v8 prior after.
Writes data/v8/priors.parquet, data/ingame_test_preds_v8.parquet (v7's file + v8 columns), data/v8/live_period_preds.parquet,
data/v8/models.pkl.  Usage: python v8/build.py
"""
import os, sys, json, pickle, importlib.util
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5'), os.path.join(SRC, 'reddit'), HERE]
os.chdir(SRC)
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff
import eval_v4 as E
import decay_inputs as DI
import buffs as BF

D = os.path.join(SRC, '../data')
SEL_END = pd.Timestamp('2026-02-01', tz='UTC')
HL_CHAMP = 30
BUFF = ['bar_on', 'eld_on']
BUFF_N = ['bar_n', 'eld_n']
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))
_s = importlib.util.spec_from_file_location('v6build', os.path.join(SRC, 'v6', 'build.py'))
B6 = importlib.util.module_from_spec(_s); _s.loader.exec_module(B6)


def champ_ext(hl):
    """pregame2's champion-table columns at half-life hl on the extended table (study + tracker games)."""
    import pregame2 as P2
    import live_period as LP
    P2.HL = float(hl)
    _, f2, _ = LP.pregame_all(LP.ext_table())
    return f2[['game_id'] + DI.CHAMP]


def weekly(P, cols, C=0.01):
    y = P.blue_win.fillna(0).astype(int).to_numpy()
    out = np.full(len(P), np.nan)
    st = pd.date_range(SPLIT, P.t_start.max() + pd.Timedelta('8D'), freq='W-MON')
    st = st.insert(0, SPLIT) if st[0] > SPLIT else st
    for t0, t1 in zip(st[:-1], st[1:]):
        cur = ((P.t_start >= t0) & (P.t_start < t1)).to_numpy()
        if cur.any():
            a = ((P.t_end < t0) & P.blue_win.notna()).to_numpy()
            sc = P.loc[a, cols].std().replace(0, 1)
            m = LogisticRegression(C=C, max_iter=5000).fit(P.loc[a, cols] / sc, y[a])
            out[cur] = m.decision_function(P.loc[cur, cols] / sc)
    return out, (m, sc)


def lp_buffs(LP):
    gids = LP.game_id.unique()
    src = ['frames60' if os.path.exists(f'{D}/frames60/{g}.parquet') else 'v5/frames60' for g in gids]
    with ProcessPoolExecutor(24) as ex:
        res = [x for x in ex.map(BF.one, gids, src) if x is not None]
    return pd.concat([r[0] for r in res], ignore_index=True)


G = {}


def _fit(args):
    m0, cols = args
    a = G['tend'] < np.datetime64(m0.tz_convert('UTC').tz_localize(None))
    X = E.X_of(G['f'][a], None, (cols, 'poly', 'ptrain'))
    return m0, tuple(cols), E.mk().fit(X, G['y'][a])


def main():
    os.makedirs(f'{D}/v8', exist_ok=True)
    cols = json.load(open(f'{D}/v7/choice.json'))['cols']
    # ---- pre-game
    P = pd.read_parquet(f'{D}/v6/prior_feats.parquet').sort_values('t_start').reset_index(drop=True)
    P['age_x_elo'] = P.elo_play_32 * np.exp(-P.patch_age / 7)
    P = P.merge(pd.read_parquet(f'{D}/v7/feat.parquet')[['game_id', 'kt', 'kl', 'kl_sd', 'bt7']], on='game_id', how='left')
    with ProcessPoolExecutor(2) as ex:
        c120, c30 = ex.map(champ_ext, (120, HL_CHAMP))
    chk = P[['game_id'] + DI.CHAMP].merge(c120, on='game_id', suffixes=('', '_r'))
    print(f'extended-table champion columns at 120 d vs prior_feats: {len(chk)} of {len(P)} games, max |diff| '
          f'{max(np.abs(chk[c] - chk[c + "_r"]).max() for c in DI.CHAMP):.2e}', flush=True)
    P = P[P.t_start >= BURN].reset_index(drop=True)
    P8 = P.drop(columns=DI.CHAMP).merge(c30, on='game_id', how='left')
    assert P8[DI.CHAMP].notna().all().all()
    pr7 = pd.read_parquet(f'{D}/v7/priors.parquet')
    P = P.merge(pr7[['game_id', 'prior_v7_m', 'pm_pre']], on='game_id', how='left')
    P8 = P8.merge(pr7[['game_id', 'prior_v7_m', 'pm_pre']], on='game_id', how='left')
    z_wk120, _ = weekly(P, cols)
    z_m30 = DI.monthly(P8, cols)
    z8u, (m_last, sc_last) = weekly(P8, cols)
    P8['prior_v8_u'] = z8u
    T = ((P.t_start >= SPLIT) & P.blue_win.notna()).to_numpy()
    y, g = P.blue_win.to_numpy(), P.match_id.to_numpy()
    print('\n== pre-game, log loss minus v7 (unblended priors) ==')
    for nm, z in (('weekly refit, 120 d', z_wk120), ('monthly, champ 30 d', z_m30), ('v8 prior: weekly + champ 30 d', z8u)):
        for lab, m in (('test', T), ('eval', T & (P.t_start >= SEL_END).to_numpy()), ('cross', T & (P.cross == 1).to_numpy())):
            a, b = ll(y[m].astype(int), sg(z[m])), ll(y[m].astype(int), sg(P.prior_v7_m.to_numpy()[m]))
            r = boot_diff(a, b, g[m])
            print(f'  {nm:30s} {lab:5s} ({m.sum():5d}) {r[0]:+.5f} [{r[1]:+.5f}, {r[2]:+.5f}]')
    w8 = B6.fit_blend(P8.assign(prior_v8_u=z8u), 'prior_v8_u')
    P8['z8'] = B6.z6_of(P8, 'prior_v8_u', w8)
    P8[['game_id', 't_start', 'cross', 'pm_pre', 'blue_win', 'prior_v7_m', 'prior_v8_u', 'z8']].to_parquet(f'{D}/v8/priors.parquet')

    # ---- in-game
    s5 = importlib.util.spec_from_file_location('v5build', os.path.join(SRC, 'v5', 'build.py'))
    B5 = importlib.util.module_from_spec(s5); s5.loader.exec_module(B5)
    f = B5.load()
    n = len(f)
    f = f.merge(pd.read_parquet(f'{D}/v8/buffs.parquet'), on=['game_id', 'ts'], how='left')
    f = f.merge(pd.read_parquet(f'{D}/game_table.parquet', columns=['game_id', 't_end']), on='game_id', how='left')
    f = f.merge(P8[['game_id', 'prior_v8_u', 'z8']], on='game_id', how='left')
    assert len(f) == n
    f['ptrain'] = np.where(f.t_start < SPLIT, f.prior_both_m, f.prior_v8_u)
    te = (f.t_start >= SPLIT).to_numpy()
    assert f.ptrain.notna().all() and f.z8[te].notna().all()
    M5 = pickle.load(open(f'{D}/v5/models.pkl', 'rb'))
    c0 = M5['cols']
    pv = pd.read_parquet(f'{D}/ingame_test_preds_v7.parquet')
    ft = f[te].reset_index(drop=True)
    assert (ft.game_id.to_numpy() == pv.game_id.to_numpy()).all()
    months = pd.date_range(SPLIT, ft.t_start.max(), freq='MS')
    LPF = pd.read_parquet(f'{D}/v7/live_period_preds.parquet')
    LPF = LPF[LPF.blue_win.notna()].reset_index(drop=True)
    lb = lp_buffs(LPF)
    n = len(LPF)
    LPF = LPF.merge(lb, on=['game_id', 'ts'], how='left').merge(P8[['game_id', 'z8']], on='game_id', how='left')
    assert len(LPF) == n
    print(f'live period: {LPF.game_id.nunique()} games; buff features for {LPF.bar_on.notna().mean():.1%} of frames; z8 for {LPF.z8.notna().mean():.1%}')
    LPF = LPF[LPF.bar_on.notna() & LPF.z8.notna()].reset_index(drop=True)
    lp_months = sorted(set(pd.date_range(SPLIT, LPF.t_start.max(), freq='MS')) - set(months))
    G.update(f=f, y=f.blue_win.astype(int).to_numpy(), tend=f.t_end.dt.tz_convert('UTC').dt.tz_localize(None).to_numpy())
    jobs = [(m0, cs) for m0 in list(months) + lp_months for cs in (c0, c0 + BUFF, c0 + BUFF_N)]
    with ProcessPoolExecutor(12) as ex:
        MD = {(m0, cs): m for m0, cs, m in ex.map(_fit, jobs)}
    mon = lambda d: d.t_start.dt.tz_convert('UTC').dt.tz_localize(None).dt.to_period('M').dt.to_timestamp().dt.tz_localize('UTC')

    def score(d, prior):
        out = {'q_v8_pre': M5['model'].predict_proba(E.X_of(d, None, (c0, 'poly', prior)))[:, 1]}
        mm = mon(d)
        for nm, cs in (('q_v8_refit', c0), ('q_v8', c0 + BUFF), ('q_v8_n', c0 + BUFF_N)):
            q = np.full(len(d), np.nan)
            for m0 in mm.unique():
                k = (mm == m0).to_numpy()
                q[k] = MD[(m0, tuple(cs))].predict_proba(E.X_of(d[k], None, (cs, 'poly', prior)))[:, 1]
            out[nm] = q
        return out
    for k, v in score(ft, 'z8').items():
        pv[k] = v
    pv['prior_v8'] = ft.z8.to_numpy()
    pv.to_parquet(f'{D}/ingame_test_preds_v8.parquet')
    for k, v in score(LPF, 'z8').items():
        LPF[k] = v
    LPF.to_parquet(f'{D}/v8/live_period_preds.parquet')
    stg = ['q_v8_pre', 'q_v8_refit', 'q_v8', 'q_v8_n']
    print('\n== in-game, per-game log loss minus q_v7 (stages cumulative) ==')
    T = pv.assign(**{c: lg(pv[c].to_numpy()) for c in stg + ['q_v7']})
    B6.report(T, stg, 'q_v7', 'all test')
    B6.report(T[T.t_start < SEL_END], stg, 'q_v7', 'selection months')
    B6.report(T[T.t_start >= SEL_END], stg, 'q_v7', 'eval months 2026-02+')
    B6.report(T[T.cross == 1], stg, 'q_v7', 'cross-league')
    L2 = LPF.assign(**{c: lg(LPF[c].to_numpy()) for c in stg + ['q_v7']})
    B6.report(L2, stg, 'q_v7', 'live period 09-17..10-11')
    ev = pv[pv.t_start >= SEL_END]
    ph = pd.cut(ev.clock / 60, [0, 10, 20, 30, 1000], right=False, labels=['0-10', '10-20', '20-30', '30+'])
    yy = ev.blue_win.astype(int).to_numpy()
    print('  by phase, eval months, per-frame LL minus q_v7:')
    for c in stg:
        dd = pd.Series(ll(yy, ev[c].to_numpy()) - ll(yy, ev.q_v7.to_numpy()))
        print(f'    {c:10s} ' + '  '.join(f'{k} {v:+.5f}' for k, v in dd.groupby(ph.to_numpy(), observed=False).mean().items()))
    last = max(months)
    pickle.dump(dict(prior=dict(cols=cols, C=0.01, hl_champ=HL_CHAMP, cadence='weekly', model=m_last, scale=sc_last), blend=w8,
                     ingame={'cols': c0 + BUFF, 'month': str(last), 'model': MD[(last, tuple(c0 + BUFF))]},
                     built=str(pd.Timestamp.now(tz='UTC'))), open(f'{D}/v8/models.pkl', 'wb'))


if __name__ == '__main__':
    main()
