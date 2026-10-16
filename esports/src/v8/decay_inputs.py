"""v8 candidate, part 1c: the decay (EWMA half-life) of v7's pre-game inputs (2026-10-16).

Several v7 inputs are already exponentially weighted averages of past games, all with a 120-day half-life chosen early on:
  champ   pregame2.HL: champ_all, champ_patch, lane_mu, pc_exp, pc_wr, side_patch, conv (decayed champion / player-champion /
          side / conversion tables)
  player  reddit/build.HL: lane_0..4, dmg_0..4 (decayed draft-residual lane gold and damage-share ratings)
  bt      v7feat.BT_HL: the Bradley-Terry sample weight 0.5 ** (age / BT_HL) -> bt7
Each family's half-life is swept over {30, 60, 120, 240, 480} days with the other two at 120; the swapped columns go into
v7's prior (monthly refit, C 0.01, v7's 55 columns). Rebuilt on data/pregame.parquet (the study's table, games to
2026-09-28), so every arm, including the 120-day reference, is rebuilt the same way; scored on test games before 2026-09-28.
Half-life chosen on the selection months (2025-07..2026-01), eval months 2026-02..09 scored once.
Usage: python v8/decay_inputs.py   (writes data/v8/inputs_hl.parquet, results/v8_decay_inputs.log)
"""
import os, sys, json
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'reddit'), os.path.join(SRC, 'v7')]
os.chdir(SRC)
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff

D = os.path.join(SRC, '../data')
SEL_END = pd.Timestamp('2026-02-01', tz='UTC')
END = pd.Timestamp('2026-09-28', tz='UTC')
HLS = (30, 60, 120, 240, 480)
CHAMP = ['champ_all', 'champ_patch', 'lane_mu', 'pc_exp', 'pc_wr', 'side_patch', 'conv']
PLAYER = [f'lane_{k}' for k in range(5)] + [f'dmg_{k}' for k in range(5)]
sig = lambda z: 1 / (1 + np.exp(-z))


def champ_feats(hl):
    import pregame2 as P2
    from sklearn.linear_model import LogisticRegression
    P2.HL = float(hl)
    p = pd.read_parquet(f'{D}/pregame.parquet')
    p['pv'] = p.patch.str.split('.').str[:2].str.join('.')
    tl = pd.concat([p[['blue_id', 'league']].rename(columns={'blue_id': 'tid'}), p[['red_id', 'league']].rename(columns={'red_id': 'tid'})])
    home = tl[~tl.league.isin(INTL)].groupby('tid').league.agg(lambda s: s.value_counts().index[0])
    p['b_league'] = p.blue_id.map(home).fillna(p.league)
    p['r_league'] = p.red_id.map(home).fillna(p.league)
    rg = pd.read_parquet(f'{D}/role_gold.parquet')
    b = p[p.t_start < BURN].merge(rg, on='game_id').dropna(subset=['gd15_0'])
    x = b[[f'gd15_{k}' for k in range(5)]].sum(1).to_numpy()[:, None] / 1000
    m = LogisticRegression().fit(x, b.blue_win.astype(int))
    a0, a1 = m.intercept_[0], m.coef_[0, 0]
    f = P2.build(p, rg, lambda gd: P2.sig(a0 + a1 * gd / 1000))
    return ('champ', hl, f[['game_id'] + CHAMP])


def player_feats(hl):
    import build as R
    R.HL = float(hl)
    p = R.load()
    ch = R.Champs(p)
    dr = R.draft_models(p, ch)
    f = R.player_pass(p, dr, R.KAL)
    return ('player', hl, f[['game_id'] + PLAYER])


def bt_feats(hl):
    import v7feat as F7
    F7.BT_HL = float(hl)
    P = pd.read_parquet(f'{D}/v6/prior_feats.parquet').sort_values('t_start').reset_index(drop=True)
    P['intl'] = P.league.isin(INTL)
    _, bt7 = F7.bt_online(P)
    return ('bt', hl, pd.DataFrame(dict(game_id=P.game_id, bt7=bt7)))


def job(a):
    kind, hl = a
    return dict(champ=champ_feats, player=player_feats, bt=bt_feats)[kind](hl)


def monthly(P, cols, C=0.01):
    from sklearn.linear_model import LogisticRegression
    y = P.blue_win.fillna(0).astype(int).to_numpy()
    out = np.full(len(P), np.nan)
    for m0 in pd.date_range(SPLIT, P.t_start.max() + pd.offsets.MonthBegin(1), freq='MS'):
        cur = ((P.t_start >= m0) & (P.t_start < m0 + pd.offsets.MonthBegin(1))).to_numpy()
        if cur.any():
            a = ((P.t_end < m0) & P.blue_win.notna()).to_numpy()
            sc = P.loc[a, cols].std().replace(0, 1)
            m = LogisticRegression(C=C, max_iter=5000).fit(P.loc[a, cols] / sc, y[a])
            out[cur] = m.decision_function(P.loc[cur, cols] / sc)
    return out


def main():
    jobs = [(k, h) for k in ('champ', 'player', 'bt') for h in HLS]
    with ProcessPoolExecutor(len(jobs)) as ex:
        res = list(ex.map(job, jobs))
    F = {(k, h): f for k, h, f in res}
    P = pd.read_parquet(f'{D}/v6/prior_feats.parquet').sort_values('t_start').reset_index(drop=True)
    P['age_x_elo'] = P.elo_play_32 * np.exp(-P.patch_age / 7)
    V7 = pd.read_parquet(f'{D}/v7/feat.parquet')[['game_id', 'kt', 'kl', 'kl_sd', 'bt7']]
    P = P.drop(columns=[c for c in ['bt7'] if c in P]).merge(V7, on='game_id', how='left')
    P = P[P.game_id.isin(set(F[('champ', 120)].game_id))]
    P = P[P.t_start >= BURN].reset_index(drop=True)
    cols = json.load(open(f'{D}/v7/choice.json'))['cols']
    chk = P[['game_id'] + CHAMP].merge(F[('champ', 120)], on='game_id', suffixes=('', '_r'))
    print(f'{len(P)} games; rebuild check at 120 d: champ max |diff| {max(np.abs(chk[c] - chk[c + "_r"]).max() for c in CHAMP):.2e}', flush=True)
    chk = P[['game_id'] + PLAYER].merge(F[('player', 120)], on='game_id', suffixes=('', '_r'))
    print(f'  player max |diff| {max(np.abs(chk[c] - chk[c + "_r"]).max() for c in PLAYER):.2e}', flush=True)
    chk = P[['game_id', 'bt7']].merge(F[('bt', 120)], on='game_id', suffixes=('', '_r'))
    print(f'  bt7 max |diff| {np.abs(chk.bt7 - chk.bt7_r).max():.2e}', flush=True)

    def with_(fam, hl):
        Q = P.copy()
        for f2, h2 in (('champ', 120), ('player', 120), ('bt', 120)):
            h = hl if f2 == fam else h2
            g = F[(f2, h)]
            Q = Q.drop(columns=[c for c in g.columns if c != 'game_id']).merge(g, on='game_id', how='left')
        return Q
    Z = {}
    for fam in ('champ', 'player', 'bt'):
        for hl in HLS:
            Q = with_(fam, hl)
            Z[(fam, hl)] = monthly(Q, cols)
            print(f'  scored {fam} {hl}', flush=True)
    T = ((P.t_start >= SPLIT) & (P.t_start < END) & P.blue_win.notna()).to_numpy()
    sel, ev = T & (P.t_start < SEL_END).to_numpy(), T & (P.t_start >= SEL_END).to_numpy()
    y, g = P.blue_win.to_numpy(), P.match_id.to_numpy()
    print(f'\ntest games {T.sum()} (selection {sel.sum()}, eval {ev.sum()}); log loss minus the 120-day rebuild (match-clustered CI)')
    out = P[['game_id']].copy()
    for fam in ('champ', 'player', 'bt'):
        base = Z[(fam, 120)]
        S = {}
        for hl in HLS:
            z = Z[(fam, hl)]
            r = []
            for m in (sel, ev):
                a, b = ll(y[m].astype(int), sig(z[m])), ll(y[m].astype(int), sig(base[m]))
                r.append((a.mean(),) + boot_diff(a, b, g[m]))
            S[hl] = r[0][0]
            out[f'{fam}_{hl}'] = z
            print(f'  {fam:6s} HL {hl:3d} d: sel {r[0][0]:.5f} ({r[0][1]:+.5f})   eval {r[1][0]:.5f} {r[1][1]:+.5f} [{r[1][2]:+.5f}, {r[1][3]:+.5f}]')
        ch = min(S, key=S.get)
        print(f'  {fam}: selection-month choice HL {ch} d')
    out.to_parquet(f'{D}/v8/inputs_hl.parquet')


if __name__ == '__main__':
    main()
