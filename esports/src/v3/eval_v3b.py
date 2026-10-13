"""v3 step 9b: minimap terms as a penalised residual on an unpenalised v2.

logit = a0 + v2 * (a1 + a2 m + a3 m^2)   [unpenalised: v2 recalibrated on these frames]
        + sum_k beta_k * z_k                 [L2 penalty lambda * |beta|^2, z standardised; k = new terms, optionally x (1, m, m^2)]
Two modes. CV mode (few train-period games): every lambda reported on 5 match-grouped folds of all minimap games (the best
row is optimistic). SPLIT mode (>= 300 train-period minimap games, the clean one): lambda picked by match-grouped 5-fold CV
inside train only, refit on all train, test games scored once.
Arms (2026-10-05, after the user's notes): float (gold not in items, feed + details), dist x up (team distance to the pit
x objective up, the "late to baron" signal), pit x up (counts), and v3 = float + dist + pit terms.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v3 as E, features as FT
from eval_pregame import ll, boot_diff
from eval_objectives import OBJ
from eval_float import add_float

D = '../data'


def fit(B, Z, y, lam):
    """B unpenalised design (with intercept), Z penalised (standardised). -> coef vector."""
    X = np.hstack([B, Z])
    nb = B.shape[1]
    pen = np.r_[np.zeros(nb), np.ones(Z.shape[1])] * lam

    def obj(w):
        z = X @ w
        lp = np.logaddexp(0, z) - y * z
        g = X.T @ (1 / (1 + np.exp(-z)) - y)
        return lp.sum() / len(y) + (pen * w * w).sum(), g / len(y) + 2 * pen * w
    w0 = np.zeros(X.shape[1]); w0[1] = 1
    return minimize(obj, w0, jac=True, method='L-BFGS-B', options=dict(maxiter=2000)).x


def design(f, cols, poly):
    m = (f.clock / 600).to_numpy()
    v = f.v2_logit.to_numpy()
    B = np.c_[np.ones(len(f)), v, v * m, v * m * m]
    if not cols:
        return B, np.zeros((len(f), 0))
    Z = f[cols].to_numpy(float)
    if poly:
        Z = np.hstack([Z, Z * m[:, None], Z * (m * m)[:, None]])
    return B, Z


def main():
    f = pd.read_parquet(f'{D}/v3/v2_logit.parquet')
    f['game_id'] = f.game_id.astype(str)
    f['sec'] = E.sec(f.ts)
    for k in ('drake', 'baron', 'elder'):
        f[f'{k}_up_gold'] = f[f'{k}_up'] * f.gold / 1000
        f[f'{k}_up_alive'] = f[f'{k}_up'] * f.alive
    M = E.load_mm().drop_duplicates(['game_id', 'sec'])
    f = f[f.game_id.isin(set(M.game_id))].merge(M, on=['game_id', 'sec'], how='left').reset_index(drop=True)
    f = add_float(f)
    f['mm_ok'] = f.mm_vis.notna().astype(float)
    MM = FT.CH + FT.WD + FT.DIST
    f[MM] = f[MM].fillna(0)
    f.loc[f.mm_ok == 0, 'dd_miss'] = 1
    f = FT.add_interactions(f)
    y = f.blue_win.astype(float).to_numpy()
    print(f'games {f.game_id.nunique()} frames {len(f)} minimap on {f.mm_ok.mean():.1%}; nan labels {np.isnan(y).sum()}')
    folds = list(GroupKFold(5).split(f, groups=f.match_id))
    gm = f.drop_duplicates('game_id').set_index('game_id').match_id
    pgs = lambda q: pd.Series(ll(y, q), index=f.game_id).groupby(level=0).mean()
    raw = pgs(1 / (1 + np.exp(-f.v2_logit.to_numpy())))

    def cvq(cols, lam, poly):
        q = np.zeros(len(f))
        for a, b in folds:
            B, Z = design(f, cols, poly)
            mu, sd = Z[a].mean(0), Z[a].std(0) + 1e-9
            Z = (Z - mu) / sd
            w = fit(B[a], Z[a], y[a], lam)
            q[b] = 1 / (1 + np.exp(-(np.hstack([B[b], Z[b]]) @ w)))
        return pgs(q)
    base = cvq([], 0, False)
    print(f'v2 as is {raw.mean():.4f}; v2 recalibrated (CV) {base.mean():.4f}')
    rows = [dict(arm='v2 as is', lam=np.nan, per_game=raw.mean()), dict(arm='v2 recal', lam=0, per_game=base.mean())]
    FLC = ['fl_team', 'fl_d1', 'fl_bad']
    DX = ['dd_miss', 'dbar_x_up', 'ddrk_x_up', 'd1bar_x_up', 'd1drk_x_up']
    arms = {'float': FLC, 'dist x up': DX, 'pit x up': FT.INTER, 'v3 = float + dist + pit': FLC + DX + FT.INTER,
            'v3 + all minimap': ['mm_ok'] + FLC + MM + FT.INTER + DX}
    LAMS = (1e-4, 1e-3, 1e-2, 1e-1)
    ntr = f[f.split == 'train'].game_id.nunique()
    if ntr >= 300:  # SPLIT mode
        tr, te = (f.split == 'train').to_numpy(), (f.split == 'test').to_numpy()
        print(f'SPLIT mode: train games {ntr}, test games {f[te].game_id.nunique()}')
        inner = list(GroupKFold(5).split(np.flatnonzero(tr), groups=f.match_id.to_numpy()[tr]))
        idx = np.flatnonzero(tr)

        def fitpred(cols, lam, poly, a, b):
            B, Z = design(f, cols, poly)
            mu, sd = Z[a].mean(0), Z[a].std(0) + 1e-9
            Z = (Z - mu) / sd
            w = fit(B[a], Z[a], y[a], lam)
            return 1 / (1 + np.exp(-(np.hstack([B[b], Z[b]]) @ w)))
        ft = f[te]
        pgt = lambda q: pd.Series(ll(y[te], q), index=ft.game_id).groupby(level=0).mean()
        gmt = ft.drop_duplicates('game_id').set_index('game_id').match_id
        b0 = pgt(fitpred([], 0, False, tr, te))
        print(f'v2 as is on test {pgs(1 / (1 + np.exp(-f.v2_logit.to_numpy())))[ft.game_id.unique()].mean():.4f}; v2 recal {b0.mean():.4f}')
        for k, cols in arms.items():
            for poly in (False, True):
                cvs = {lam: np.mean([ll(y[idx[bb]], fitpred(cols, lam, poly, idx[aa], idx[bb])).mean() for aa, bb in inner]) for lam in LAMS}
                lam = min(cvs, key=cvs.get)
                pg = pgt(fitpred(cols, lam, poly, tr, te))
                d, lo, hi = boot_diff(pg.to_numpy(), b0.reindex(pg.index).to_numpy(), gmt.reindex(pg.index).to_numpy())
                print(f'{k:24s} poly={poly!s:5} lam*={lam:<6} test {pg.mean():.4f}  vs v2 recal {d:+.4f} [{lo:+.4f}, {hi:+.4f}]', flush=True)
                rows.append(dict(mode='split', arm=k, poly=poly, lam=lam, per_game=pg.mean(), d=d, lo=lo, hi=hi))
    else:
        for k, cols in arms.items():
            for poly in (False, True):
                for lam in LAMS:
                    pg = cvq(cols, lam, poly)
                    d, lo, hi = boot_diff(pg.to_numpy(), base.reindex(pg.index).to_numpy(), gm.reindex(pg.index).to_numpy())
                    print(f'{k:24s} poly={poly!s:5} lam={lam:<6} {pg.mean():.4f}  vs v2 recal {d:+.4f} [{lo:+.4f}, {hi:+.4f}]', flush=True)
                    rows.append(dict(mode='cv', arm=k, poly=poly, lam=lam, per_game=pg.mean(), d=d, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(f'{D}/../results/v3_eval_b.csv', index=False)


if __name__ == '__main__':
    main()
