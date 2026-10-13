"""Freeze v3 for the live tracker (2026-10-05). Two residual stages on the frozen v2 logit:

  A  float:   z_A = a0 + v2 (a1 + a2 m + a3 m^2) + sum_k b_k x_k (1, m, m^2),  x = fl_team, fl_d1, fl_bad  (all 20k feed games;
              v2 logits out-of-fold on train, from the train fit on test; L2 1e-4 on the standardised new terms)
  B  minimap: z = z_A + sum_j g_j (y_j - mean_j) / sd_j   (z_A a fixed offset),  y = dist x up (dd_miss, dbar/ddrk/d1bar/d1drk_x_up) and pit x up (drk/wdrk/bar/wbar_x_up)
              (games with parsed minimap frames, both periods; L2 0.1, no time terms: the most stable setting in eval_v3b's CV)
Neither stage beat v2 on resolution log loss out of sample (A −0.0003 borderline, B ns); this is an exploratory live signal.
Writes ~/fyp-pmlive/lollive/data/v3/v3_model.pkl: coefficients, standardisation, feature names, float glitch limits.
"""
import os, sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v3 as E, features as FT
from eval_v3b import fit
from eval_float import add_float

D = '../data'
OUT = os.path.expanduser('~/fyp-pmlive/lollive/data/v3')
FLC = ['fl_team', 'fl_d1', 'fl_bad']
MMC = ['dd_miss', 'dbar_x_up', 'ddrk_x_up', 'd1bar_x_up', 'd1drk_x_up'] + FT.INTER


def main():
    f = pd.read_parquet(f'{D}/v3/v2_logit.parquet')
    f['game_id'] = f.game_id.astype(str)
    f['sec'] = E.sec(f.ts)
    f = add_float(f)
    # glitch limits used by add_float, saved for live use
    b = (f.clock // 300).clip(upper=9)
    lim = (2 * pd.concat([f.fl_b, f.fl_r]).groupby(pd.concat([b, b])).quantile(0.99)).to_dict()
    y = f.blue_win.astype(float).to_numpy()
    m = (f.clock / 600).to_numpy()
    v = f.v2_logit.to_numpy()
    B = np.c_[np.ones(len(f)), v, v * m, v * m * m]
    Z = f[FLC].to_numpy(float)
    Z = np.hstack([Z, Z * m[:, None], Z * (m * m)[:, None]])
    muA, sdA = Z.mean(0), Z.std(0) + 1e-9
    wA = fit(B, (Z - muA) / sdA, y, 1e-4)
    zA = np.hstack([B, (Z - muA) / sdA]) @ wA
    f['zA'] = zA
    print('stage A coefs (base a0..a3, then float x (1,m,m2)):', wA.round(4).tolist())
    # stage B on minimap games
    M = E.load_mm().drop_duplicates(['game_id', 'sec'])
    g = f[f.game_id.isin(set(M.game_id))].merge(M, on=['game_id', 'sec'], how='left').reset_index(drop=True)
    g['mm_ok'] = g.mm_vis.notna().astype(float)
    MM = FT.CH + FT.WD + FT.DIST
    g[MM] = g[MM].fillna(0)
    g.loc[g.mm_ok == 0, 'dd_miss'] = 1
    g = FT.add_interactions(g)
    yb = g.blue_win.astype(float).to_numpy()
    # zA is a fixed offset (slope 1, no intercept): a free intercept/slope here would recalibrate v2 on these 463 mostly
    # 2025-26 games (it came out at c0 = -0.17), which is not what the minimap terms are for
    from scipy.optimize import minimize
    Zb = g[MMC].to_numpy(float)
    muB, sdB = Zb.mean(0), Zb.std(0) + 1e-9
    Zs = (Zb - muB) / sdB
    off = g.zA.to_numpy()

    def obj(w):
        z = off + Zs @ w
        return (np.logaddexp(0, z) - yb * z).mean() + 0.1 * (w * w).sum(), Zs.T @ (1 / (1 + np.exp(-z)) - yb) / len(yb) + 0.2 * w
    wB = minimize(obj, np.zeros(Zs.shape[1]), jac=True, method='L-BFGS-B').x
    print(f'stage B on {g.game_id.nunique()} games, {len(g)} frames; coefs per SD:', dict(zip(MMC, wB.round(4))))
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(dict(FLC=FLC, MMC=MMC, wA=wA, muA=muA, sdA=sdA, wB=wB, muB=muB, sdB=sdB, fl_lim=lim,
                     n_games_B=int(g.game_id.nunique()), built='2026-10-05'), open(f'{OUT}/v3_model.pkl', 'wb'))
    print('written', f'{OUT}/v3_model.pkl')


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


def build_takers():
    """Logged-only side models: P(blue takes the drake/elder | baron that is up), feed + minimap (eval_taker.py's 'feed + both'),
    fit on all minimap games. Saved next to v3_model.pkl as v3_takers.pkl."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import eval_taker as T
    f = pd.read_parquet(f'{D}/v3/v2_logit.parquet')
    f['game_id'] = f.game_id.astype(str)
    f['sec'] = E.sec(f.ts)
    M = E.load_mm().drop_duplicates(['game_id', 'sec'])
    f = f[f.game_id.isin(set(M.game_id))].merge(M, on=['game_id', 'sec'], how='inner')
    ing = pd.read_parquet(f'{D}/ingame.parquet', columns=['game_id', 'ts', 'kills'])
    ing['game_id'] = ing.game_id.astype(str)
    ing['sec'] = E.sec(ing.ts.dt.tz_convert('UTC') if ing.ts.dt.tz is not None else ing.ts.dt.tz_localize('UTC'))
    f = f.merge(ing[['game_id', 'sec', 'kills']], on=['game_id', 'sec'], how='left')
    f = f.merge(pd.concat([T.takers(g) for g in f.game_id.unique()]), on=['game_id', 'sec'], how='left')
    f['gold'] = f.gold / 1000
    base = ['v2_logit', 'gold', 'alive', 'kills']
    out = {}
    for obj, up, lab, cols in [('drake', (f.drake_up + f.elder_up) > 0, 'next_drake_blue', ['mm_drk', 'mm_wdrk', 'mm_adv', 'mm_vis', 'd_drk', 'd1_drk', 'dd_miss']),
                               ('baron', f.baron_up > 0, 'next_baron_blue', ['mm_bar', 'mm_wbar', 'mm_adv', 'mm_vis', 'd_bar', 'd1_bar', 'dd_miss'])]:
        g = f[up & f[lab].notna()]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(g[base + cols], g[lab].astype(int))
        out[obj] = dict(model=m, cols=base + cols, n=len(g))
        print(f'taker {obj}: {len(g)} frames')
    pickle.dump(out, open(f'{OUT}/v3_takers.pkl', 'wb'))


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'takers':
    build_takers()
