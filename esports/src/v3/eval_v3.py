"""v3 step 9: does the minimap add to v2? Residual logistic on top of v2's out-of-fold logit, on frames with a parsed minimap.

Arms (each term x (1, m, m^2), m = minutes / 10; StandardScaler + LogisticRegression C=1; fit on train-split games):
  v2            v2 logit only (v2 recalibrated on exactly these frames: the fair baseline)
  v2+obj        + objective x (gold, alive) and soul terms (eval_objectives; feed only)
  v3            + minimap features (features.py: champion and ward positions, pit presence x objective up)
Frames where the minimap was not found keep the minimap features at 0 with an indicator (so v3 = v2 there).
Per-game log loss on test games, match-clustered bootstrap CIs. Writes results/v3_eval.csv and data/v3/v3_test_preds.parquet.
"""
import os, sys, glob
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pregame import ll, boot_diff
from eval_objectives import OBJ
import features as FT

D = '../data'


def sec(ts):
    return ((ts - pd.Timestamp(0, tz='UTC')).dt.total_seconds() // 1).astype('int64')


def load_mm():
    P = pd.concat([pd.read_parquet(f) for f in glob.glob(f'{D}/v3/parsed2/*.parquet')], ignore_index=True)
    P = P[P['loc'] >= 0.45].reset_index(drop=True)
    F = FT.encode(P)
    F['game_id'], F['sec'] = P.game_id.astype(str), sec(P.ts)
    return F.dropna(subset=['mm_vis'])


def poly(f, cols):
    m = f.clock / 600
    X = {}
    for c in cols:
        X[c], X[c + '_m'], X[c + '_m2'] = f[c], f[c] * m, f[c] * m * m
    return pd.DataFrame(X)


def main():
    f = pd.read_parquet(f'{D}/v3/v2_logit.parquet')
    f['game_id'] = f.game_id.astype(str)
    f['sec'] = sec(f.ts)
    for k in ('drake', 'baron', 'elder'):
        f[f'{k}_up_gold'] = f[f'{k}_up'] * f.gold / 1000
        f[f'{k}_up_alive'] = f[f'{k}_up'] * f.alive
    M = load_mm()
    f = f[f.game_id.isin(set(M.game_id))].merge(M, on=['game_id', 'sec'], how='left')
    f['mm_ok'] = f.mm_vis.notna().astype(float)
    MM = FT.CH + FT.WD
    f[MM] = f[MM].fillna(0)
    f = FT.add_interactions(f)
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.split == 'train').to_numpy(), (f.split == 'test').to_numpy()
    print(f'games train {f.game_id[tr].nunique()} test {f.game_id[te].nunique()}; frames {len(f)}; minimap found on {f.mm_ok.mean():.1%}')
    arms = {'v2': ['v2_logit'], 'v2+obj': ['v2_logit'] + OBJ, 'v3': ['v2_logit', 'mm_ok'] + OBJ + MM + FT.INTER,
            'v3 minimap only': ['v2_logit', 'mm_ok'] + MM + FT.INTER}
    Q, coefs = {}, {}
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    cv = f.game_id[tr].nunique() < 200  # too few train-period games with a minimap: match-grouped CV inside the test period
    if cv:
        from sklearn.model_selection import GroupKFold
        te = np.ones(len(f), bool)
        folds = list(GroupKFold(5).split(f, groups=f.match_id))
        print('CV mode: 5 folds grouped by match, all minimap games (v2_logit is out-of-sample for all of them)')
    for k, cols in arms.items():
        X = poly(f, cols)
        if cv:
            q = np.zeros(len(f))
            for a, b in folds:
                q[b] = mk().fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
            Q[k] = q
            m = mk().fit(X, y)
        else:
            m = mk().fit(X[tr], y[tr])
            Q[k] = m.predict_proba(X[te])[:, 1]
        coefs[k] = pd.Series(m[-1].coef_[0] / m[0].scale_, index=X.columns)
    ft = f[te].reset_index(drop=True)
    gm = ft.drop_duplicates('game_id').set_index('game_id').match_id
    pg = {k: pd.Series(ll(y[te], q), index=ft.game_id).groupby(level=0).mean() for k, q in Q.items()}
    rows = []
    for k in Q:
        d, lo, hi = boot_diff(pg[k].to_numpy(), pg['v2'].reindex(pg[k].index).to_numpy(), gm.reindex(pg[k].index).to_numpy())
        rows.append(dict(model=k, per_game=pg[k].mean(), d_vs_v2=d, lo=lo, hi=hi))
        print(f'{k:16s} {pg[k].mean():.4f}  vs v2 {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
    # where: frames with an objective up and someone at the pit
    for lab, s in [('drake/elder up', (ft.drake_up + ft.elder_up) > 0), ('baron up', ft.baron_up > 0),
                   ('objective up & pit contested', ((ft.drk_x_up != 0) | (ft.bar_x_up != 0)))]:
        s = s.to_numpy()
        print(f'  {lab:30s} n {s.sum():6d}: v3 - v2 frame log loss {ll(y[te][s], Q["v3"][s]).mean() - ll(y[te][s], Q["v2"][s]).mean():+.4f}')
    print('v3 minimap coefficients (per unit, at m = 0 / 1 / 2 i.e. 0 / 10 / 20 min):')
    c = coefs['v3']
    print(pd.DataFrame({t: {k: c[k] + c[k + '_m'] * t + c[k + '_m2'] * t * t for k in MM + FT.INTER} for t in (0, 1, 2)}).round(3).to_string())
    pd.DataFrame(rows).to_csv(f'{D}/../results/v3_eval.csv', index=False)
    ft[['game_id', 'ts', 'clock']].assign(**{f'q_{k}': v for k, v in Q.items()}).to_parquet(f'{D}/v3/v3_test_preds.parquet')


if __name__ == '__main__':
    main()
