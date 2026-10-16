"""v7 = v6 with the v7 prior (v7/prior.py: the union prior with the Bradley-Terry rating updated after every game, plus the
team + league Kalman ratings on dominance-weighted results). Everything else is v6's recipe, refit on the new prior:
  z7  = z_v7 + has_pm * (w0 + w1 cross) * (logit pm_pre - z_v7), (w0, w1) on the selection months (SPLIT .. SEL_END)
  q_v7 = v5's frozen in-game logistic with z7 as its prior (v6's persistence step is re-checked by the same CV)
  q_v7m = v7 combined with the live mid per phase (fit in v7/backtest_v7.py on the archive selection months)
Writes data/v7/models.pkl, data/ingame_test_preds_v7.parquet (v6's file + q_v7, prior_v7), data/v7/live_period_preds.parquet.
Usage: python v7/build.py
"""
import os, sys, pickle, json, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5')]
from eval_pregame import SPLIT, ll, boot_diff

_s = importlib.util.spec_from_file_location('v6build', os.path.join(SRC, 'v6', 'build.py'))
B6 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(B6)
D = os.path.join(SRC, '../data')
lg, sg = B6.lg, B6.sg
ZX = 'prior_v7_m'


def main():
    M5 = pickle.load(open(f'{D}/v5/models.pkl', 'rb'))
    PR = pd.read_parquet(f'{D}/v7/priors.parquet')
    w = B6.fit_blend(PR, ZX)
    PR['z6'] = B6.z6_of(PR, ZX, w)  # v6's helpers call the blended prior z6; here it is z7
    pcols = ['game_id', ZX, 'z6']

    pv = pd.read_parquet(f'{D}/ingame_test_preds_v6.parquet')
    _s5 = importlib.util.spec_from_file_location('v5build', os.path.join(SRC, 'v5', 'build.py'))
    B5 = importlib.util.module_from_spec(_s5)
    _s5.loader.exec_module(B5)
    f = B5.load()
    te = f[f.t_start >= SPLIT].reset_index(drop=True)
    assert (te.game_id.to_numpy() == pv.game_id.to_numpy()).all()
    te = te.merge(PR[pcols + ['pm_pre', 'cross']], on='game_id', how='left')
    assert len(te) == len(pv) and te.z6.notna().all(), te.z6.isna().sum()
    te['zx'] = B6.v5_logit(M5, te, ZX)
    te['zb'] = B6.v5_logit(M5, te, 'z6')
    sel = te[te.t_start < B6.SEL_END]
    gp = B6.fit_persist(sel, ZX)
    te['zp'] = te.zb + B6.persist_X(te, ZX) @ gp
    pv['q_v7_x'] = sg(te.zx.to_numpy())
    pv['q_v7'] = sg(te.zp.to_numpy())
    pv['prior_v7'] = te.z6.to_numpy()
    pv.to_parquet(f'{D}/ingame_test_preds_v7.parquet')
    print('== test split, per-game log loss (v7_x = v7 prior without the market blend) ==')
    T = pv.assign(**{c: lg(pv[c].to_numpy()) for c in ['q_v7_x', 'q_v7', 'q_v6', 'q_v5']})
    for base in ('q_v6', 'q_v5'):
        B6.report(T, ['q_v7_x', 'q_v7'] + (['q_v6'] if base == 'q_v5' else []), base, 'all test')
    B6.report(T[T.t_start >= B6.SEL_END], ['q_v7_x', 'q_v7'], 'q_v6', 'eval months 2026-02+')
    B6.report(T[T.cross == 1], ['q_v7_x', 'q_v7'], 'q_v6', 'cross-league')
    B6.report(T[T.league.isin(['worlds', 'msi', 'first_stand', 'ewc_lol', 'esports_world_cup'])], ['q_v7_x', 'q_v7'], 'q_v6', 'international')

    F = pd.read_parquet(f'{D}/v6/live_period_preds.parquet')
    F = F.drop(columns=['z6']).merge(PR[pcols], on='game_id', how='left')
    print(f'live period: {F.game_id.nunique()} games, prior_v7 present {F[ZX].notna().mean():.1%}')
    F['zx'] = B6.v5_logit(M5, F, ZX)
    F['zb'] = B6.v5_logit(M5, F, 'z6')
    F['zp'] = F.zb + B6.persist_X(F, ZX) @ gp
    F['q_v7_x'] = sg(F.zx.to_numpy())
    F['q_v7'] = sg(F.zp.to_numpy())
    F['prior_v7'] = F.z6
    F.drop(columns=['zx', 'zb', 'zp']).to_parquet(f'{D}/v7/live_period_preds.parquet')
    T = F.assign(**{c: lg(F[c].to_numpy()) for c in ['q_v7_x', 'q_v7', 'q_v6', 'q_v5']})
    B6.report(T, ['q_v7_x', 'q_v7', 'q_v5'], 'q_v6', 'live period, all games')
    B6.report(T[T.pm_pre.notna()], ['q_v7_x', 'q_v7'], 'q_v6', 'live period, games with a captured pre-game price')
    ch = json.load(open(f'{D}/v7/choice.json'))
    pickle.dump(dict(v5=M5, prior=ch, blend=w, persist=gp, split=str(SPLIT), sel_end=str(B6.SEL_END),
                     built=str(pd.Timestamp.now(tz='UTC'))), open(f'{D}/v7/models.pkl', 'wb'))


if __name__ == '__main__':
    main()
