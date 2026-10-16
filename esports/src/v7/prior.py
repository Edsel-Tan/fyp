"""v7 pre-game prior: v5/v6's union prior plus the v7 rating features (v7feat.py), monthly refit (2026-10-16).

Steps:
  0 per-game g_area (mean tanh(gold diff / 5k)) for every game with frames: the study's 60 s frames (data/ingame.parquet),
    v5/live_period's new-game frames, else the tracker's saved feed (lollive data/frames60)
  1 v7feat.features on the extended table (v6/prior_feats.parquet's games); check bt_w == the study's bt
  2 Kalman settings: small grid, each scored as union + (kt, kl, kl_sd) on the validation slice (fit < VAL0, score VAL0..SPLIT)
  3 feature arms on the validation slice (C 0.01): union; bt -> bt7; + Kalman; + day_dom (and x cross); combinations.
    The chosen arm is the one with the best validation LL. Then every arm on the test months (monthly refit), once.
Writes data/v7/feat.parquet, data/v7/priors.parquet (prior_union_m, prior_v7_m, pm_pre, ...), data/v7/choice.json.
Usage: python v7/prior.py
"""
import os, sys, json, glob, itertools
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, HERE, os.path.join(SRC, 'reddit'), os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5'), os.path.join(SRC, 'v6')]
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff
import v7feat as F7

D = os.path.join(SRC, '../data')
LLD = os.path.expanduser('~/fyp-pmlive/lollive/data')
VAL0 = SPLIT - pd.Timedelta('91D')
sig = lambda z: 1 / (1 + np.exp(-z))


def g_areas(gids):
    f = pd.read_parquet(f'{D}/ingame.parquet', columns=['game_id', 'clock', 'gold'])
    out = {g: F7.gold_area(x.clock.to_numpy(), x.gold.to_numpy()) for g, x in f.groupby('game_id')}
    for g in gids:
        if g in out:
            continue
        for fp in (f'{D}/v5/frames60/{g}.parquet', f'{LLD}/frames60/{g}.parquet'):
            if os.path.exists(fp):
                out[g] = F7.gold_area_feed(pd.read_parquet(fp, columns=['ts', 'state', 'b_gold', 'r_gold']))
                break
    return pd.Series(out, name='g_area')


def load():
    P = pd.read_parquet(f'{D}/v6/prior_feats.parquet').sort_values('t_start').reset_index(drop=True)
    P['age_x_elo'] = P.elo_play_32 * np.exp(-P.patch_age / 7)
    P['intl'] = P.league.isin(INTL)
    ga = g_areas(P.game_id)
    P['g_area'] = P.game_id.map(ga)
    return P


def fitpred(P, cols, C, a):
    y = P.blue_win.fillna(0).astype(int).to_numpy()
    sc = P.loc[a, cols].std().replace(0, 1)
    m = LogisticRegression(C=C, max_iter=5000).fit(P.loc[a, cols] / sc, y[a])
    return m.decision_function(P[cols] / sc)


def val_ll(P, cols, C=0.01):
    fit = ((P.t_start >= BURN) & (P.t_end < VAL0) & P.blue_win.notna()).to_numpy()
    va = ((P.t_start >= VAL0) & (P.t_start < SPLIT)).to_numpy()
    z = fitpred(P, cols, C, fit)[va]
    y = P.blue_win[va].astype(int).to_numpy()
    L = ll(y, sig(z))
    cr = P.cross.to_numpy()[va] == 1
    return L.mean(), L[cr].mean()


def _kal(args):
    P, k = args
    return F7.kalman(P, k)


def main():
    import eval as RE
    import importlib.util
    sp_ = importlib.util.spec_from_file_location('v6prior', os.path.join(SRC, 'v6', 'prior.py'))
    V6 = importlib.util.module_from_spec(sp_); sp_.loader.exec_module(V6)
    os.makedirs(f'{D}/v7', exist_ok=True)
    P = load()
    print(f'{len(P)} games; g_area for {P.g_area.notna().mean():.1%}', flush=True)
    UN = RE.V4P + RE.POST
    F = F7.features(P)
    P = pd.concat([P, F.drop(columns='game_id')], axis=1)
    print(f'bt_w vs the study bt: max |diff| {np.abs(P.bt_w - P.bt).max():.2e}; corr(bt7, bt) {np.corrcoef(P.bt7, P.bt)[0, 1]:.4f}; '
          f'surprise E[g_area | kp] fit on games before {F7.E_END:%Y-%m}', flush=True)

    # 2 Kalman settings: chosen by the validation grid in results/v7_kgrid.log (F7.KAL7); the win/loss-only filter (lam 1) is an ablation arm
    best = dict(F7.KAL7)
    print(f'Kalman settings {best}', flush=True)
    K1 = F7.kalman(P, dict(lam=1.0))
    P[['kt1', 'kl1', 'kl_sd1', 'kp1']] = K1.to_numpy()
    S, b = F7.surprise(P, P.kp.to_numpy())
    P[['day_dom', 'ser_dom', 'd2_dom']] = S.to_numpy()
    P[['game_id'] + F7.FEATS + ['bt_w', 'g_area']].to_parquet(f'{D}/v7/feat.parquet')  # every game (the tracker's g_area seed)
    P = P[P.t_start >= BURN].reset_index(drop=True)  # features use the whole table (rating burn-in); fits start at BURN (v6's protocol, the tracker's)
    for c in ('day_dom', 'kl', 'd2_dom'):
        P[f'x_{c}'] = P[c] * P.cross
    P['bt_old'] = P.bt
    UN7 = [c if c != 'bt' else 'bt7' for c in UN]
    KF = ['kt', 'kl', 'kl_sd']
    arms = {'union (v5/v6)': UN, 'bt -> bt7': UN7, '+ Kalman': UN + KF, '+ Kalman x cross': UN + KF + ['x_kl'],
            '+ day_dom': UN + ['day_dom'], '+ day_dom x cross': UN + ['day_dom', 'x_day_dom'], '+ ser_dom': UN + ['ser_dom'],
            '+ d2_dom': UN + ['d2_dom'],
            'bt7 + Kalman': UN7 + KF, 'bt7 + Kalman (win/loss updates)': UN7 + ['kt1', 'kl1', 'kl_sd1'], 'bt7 + Kalman + day_dom': UN7 + KF + ['day_dom'],
            'bt7 + Kalman + day_dom x cross': UN7 + KF + ['day_dom', 'x_day_dom'],
            'bt + bt7 + Kalman + day_dom': UN + ['bt7'] + KF + ['day_dom']}
    V = {k: val_ll(P, v) for k, v in arms.items()}
    print('\nvalidation LL (fit < 2025-04-01, score 2025-04..06):')
    for k, (a, c) in V.items():
        print(f'  {k:32s} {a:.5f}  cross {c:.5f}')
    v7 = [k for k in arms if k.startswith('bt7 + Kalman') and 'win/loss' not in k]
    choice = min(v7, key=lambda k: V[k][0])
    print(f'chosen v7 arm (best validation LL among the bt7 + Kalman arms): {choice}')

    T = ((P.t_start >= SPLIT) & P.blue_win.notna()).to_numpy()
    y = P.blue_win[T].astype(int).to_numpy()
    g = P.match_id[T].to_numpy()
    cr = P.cross[T].to_numpy() == 1
    intl = P.intl[T].to_numpy()
    k2 = (P.number[T] > 1).to_numpy()
    Z = {k: V6.monthly(P, v, 0.01)[T] for k, v in arms.items()}
    base = ll(y, sig(Z['union (v5/v6)']))
    print(f'\ntest months 2025-07..{P.t_start.max():%Y-%m-%d}: {T.sum()} games ({cr.sum()} cross-league, {intl.sum()} international, {k2.sum()} game 2+); LL minus union:')
    for k, z in Z.items():
        a = ll(y, sig(z))
        f = lambda m: boot_diff(a[m], base[m], g[m])
        r = [f(m) for m in (np.ones(len(a), bool), cr, intl, k2)]
        print(f'  {k:32s} {a.mean():.4f}  ' + '  '.join(f'{nm} {x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]' for nm, x in zip(('all', 'cross', 'intl', 'game2+'), r)))
    P['prior_union_m'] = V6.monthly(P, UN, 0.01)
    P['prior_v7_m'] = V6.monthly(P, arms[choice], 0.01)
    P = P.merge(V6.pm_pre_all(), on='game_id', how='left')
    keep = ['game_id', 'match_id', 'number', 't_start', 't_end', 'league', 'b_league', 'r_league', 'cross', 'intl', 'src', 'blue_win',
            'n_min_games', 'prior_union_m', 'prior_v7_m', 'pm_pre', 'kt', 'kl', 'kl_sd', 'kp', 'bt7', 'bt_old', 'day_dom', 'g_area']
    P[keep].to_parquet(f'{D}/v7/priors.parquet')
    json.dump(dict(kalman=best, arm=choice, cols=arms[choice], surprise_coef=list(map(float, b)), C=0.01),
              open(f'{D}/v7/choice.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
