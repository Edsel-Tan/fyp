"""Overfitting check of the frozen live v3 (build_live_v3.py), 2026-10-05.
Stage A (v2 recal + float): in-sample vs train->test. Stage B (minimap on 463 games): in-sample vs 5-fold match-grouped CV.
Scores are per-game mean log loss; 'v2' is the plain out-of-sample v2 logit."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.optimize import minimize
import eval_v3 as E, features as FT
from eval_v3b import fit
from eval_float import add_float
from eval_pregame import SPLIT
from build_live_v3 import FLC, MMC

D = '../data'


def pgl(y, z, gid):
    l = np.logaddexp(0, z) - y * z
    return pd.Series(l).groupby(gid).mean().mean()


f = pd.read_parquet(f'{D}/v3/v2_logit.parquet')
f['game_id'] = f.game_id.astype(str)
f['sec'] = E.sec(f.ts)
f = add_float(f).reset_index(drop=True)
y = f.blue_win.astype(float).to_numpy()
m = (f.clock / 600).to_numpy()
v = f.v2_logit.to_numpy()
gid = f.game_id.to_numpy()
B = np.c_[np.ones(len(f)), v, v * m, v * m * m]
Z = f[FLC].to_numpy(float)
Z = np.hstack([Z, Z * m[:, None], Z * (m * m)[:, None]])
tr = (f.t_start < SPLIT).to_numpy(); te = ~tr
mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
wtr = fit(B[tr], (Z[tr] - mu) / sd, y[tr], 1e-4)
zA_tr = np.hstack([B, (Z - mu) / sd]) @ wtr
muA, sdA = Z.mean(0), Z.std(0) + 1e-9
wall = fit(B, (Z - muA) / sdA, y, 1e-4)
zA_all = np.hstack([B, (Z - muA) / sdA]) @ wall
print(f'stage A ({len(wall)} params, {f.game_id.nunique()} games)')
for lab, s in [('train', tr), ('test', te)]:
    print(f'  {lab:5s}: v2 {pgl(y[s], v[s], gid[s]):.4f}  A fit-on-train {pgl(y[s], zA_tr[s], gid[s]):.4f}  A fit-on-all (live) {pgl(y[s], zA_all[s], gid[s]):.4f}')

# stage B on minimap games, zA (live version) as offset
f['zA'] = zA_all
M = E.load_mm().drop_duplicates(['game_id', 'sec'])
g = f[f.game_id.isin(set(M.game_id))].merge(M, on=['game_id', 'sec'], how='left').reset_index(drop=True)
g['mm_ok'] = g.mm_vis.notna().astype(float)
MM = FT.CH + FT.WD + FT.DIST
g[MM] = g[MM].fillna(0)
g.loc[g.mm_ok == 0, 'dd_miss'] = 1
g = FT.add_interactions(g)
yb = g.blue_win.astype(float).to_numpy()
Zb = g[MMC].to_numpy(float)
off = g.zA.to_numpy()
gg = g.game_id.to_numpy()


def fitB(ix, lam=0.1):
    mu, sd = Zb[ix].mean(0), Zb[ix].std(0) + 1e-9
    Zs = (Zb[ix] - mu) / sd

    def obj(w):
        z = off[ix] + Zs @ w
        return (np.logaddexp(0, z) - yb[ix] * z).mean() + lam * (w * w).sum(), Zs.T @ (1 / (1 + np.exp(-z)) - yb[ix]) / ix.sum() + 2 * lam * w
    w = minimize(obj, np.zeros(Zs.shape[1]), jac=True, method='L-BFGS-B').x
    return lambda jx: off[jx] + ((Zb[jx] - mu) / sd) @ w


mids = g.match_id.unique()
k = g.match_id.map(pd.Series(np.random.default_rng(0).integers(0, 5, len(mids)), index=mids)).to_numpy()
print(f'stage B ({len(MMC)} params, {g.game_id.nunique()} games, {len(g)} frames)')
allix = np.ones(len(g), bool)
for lam in (0.1, 0.01, 0.0):
    zin = fitB(allix, lam)(allix)
    zcv = np.empty(len(g))
    for j in range(5):
        zcv[k == j] = fitB(k != j, lam)(k == j)
    print(f'  L2 {lam:<5}: v2 {pgl(yb, g.v2_logit.to_numpy(), gg):.4f}  stage A only {pgl(yb, off, gg):.4f}  '
          f'B in-sample {pgl(yb, zin, gg):.4f}  B 5-fold CV {pgl(yb, zcv, gg):.4f}')
