"""Extends v8/decay_inputs.py's champion-table half-life grid below its 30-day edge (7, 15, 20, 30 d vs 120 d); same protocol.
Usage: python v8/decay_inputs_champ.py   (results/v8_decay_inputs_champ.log, data/v8/inputs_hl_champ.parquet)"""
import os, sys, json
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decay_inputs as DI
from eval_pregame import BURN, SPLIT, ll, boot_diff

HLS = (7, 15, 20, 30, 120)


def main():
    with ProcessPoolExecutor(len(HLS)) as ex:
        F = {h: f for _, h, f in ex.map(DI.champ_feats, HLS)}
    D = DI.D
    P = pd.read_parquet(f'{D}/v6/prior_feats.parquet').sort_values('t_start').reset_index(drop=True)
    P['age_x_elo'] = P.elo_play_32 * np.exp(-P.patch_age / 7)
    P = P.merge(pd.read_parquet(f'{D}/v7/feat.parquet')[['game_id', 'kt', 'kl', 'kl_sd', 'bt7']], on='game_id', how='left')
    P = P[P.game_id.isin(set(F[120].game_id)) & (P.t_start >= BURN)].reset_index(drop=True)
    cols = json.load(open(f'{D}/v7/choice.json'))['cols']
    Z = {h: DI.monthly(P.drop(columns=DI.CHAMP).merge(F[h], on='game_id', how='left'), cols) for h in HLS}
    T = ((P.t_start >= SPLIT) & (P.t_start < DI.END) & P.blue_win.notna()).to_numpy()
    sel, ev = T & (P.t_start < DI.SEL_END).to_numpy(), T & (P.t_start >= DI.SEL_END).to_numpy()
    y, g = P.blue_win.to_numpy(), P.match_id.to_numpy()
    out = P[['game_id']].copy()
    for h in HLS:
        r = []
        for m in (sel, ev):
            a, b = ll(y[m].astype(int), DI.sig(Z[h][m])), ll(y[m].astype(int), DI.sig(Z[120][m]))
            r.append((a.mean(),) + boot_diff(a, b, g[m]))
        out[f'champ_{h}'] = Z[h]
        print(f'  champ HL {h:3d} d: sel {r[0][0]:.5f} ({r[0][1]:+.5f})   eval {r[1][0]:.5f} {r[1][1]:+.5f} [{r[1][2]:+.5f}, {r[1][3]:+.5f}]', flush=True)
    out.to_parquet(f'{D}/v8/inputs_hl_champ.parquet')
    for h in HLS:
        F[h].to_parquet(f'{D}/v8/champ_feats_hl{h}.parquet')


if __name__ == '__main__':
    main()
