"""v3 step 10: does the minimap predict who takes the objective that is up (the swing itself), beyond the feed?

Frames with drake / elder (or baron) up; label = the side that takes the next one (blue 1, red 0), from the cumulative feed
counts in later frames (frames where nobody takes it before the game ends are dropped). Base: logistic on v2_logit, gold,
alive, kills (blue - red); + minimap: mm_drk / mm_bar (champions at the pit), mm_wdrk / mm_wbar (wards at the pit), mm_adv.
5-fold match-grouped CV on the minimap games; per-frame log loss and AUC, game-clustered CI.
"""
import os, sys, glob
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_v3 as E
from eval_pregame import ll, boot_diff
from objectives import counts

D = '../data'


def takers(gid):
    d = pd.read_parquet(f'{D}/frames60/{gid}.parquet')
    d = d[d.state.isin(['in_game', 'paused', 'finished'])]
    d = d.assign(_t=pd.to_datetime(d.ts, format='ISO8601')).sort_values('_t').drop_duplicates('_t')
    d = d[(d.b_gold + d.r_gold) > 0].reset_index(drop=True)
    lb = d.b_drakes.fillna('').str.split(',').map(lambda l: len([x for x in l if x])).to_numpy()
    lr = d.r_drakes.fillna('').str.split(',').map(lambda l: len([x for x in l if x])).to_numpy()
    bb, br = d.b_barons.to_numpy(), d.r_barons.to_numpy()
    nd, nb = np.full(len(d), np.nan), np.full(len(d), np.nan)
    for i in range(len(d)):
        j = next((k for k in range(i + 1, len(d)) if lb[k] + lr[k] > lb[i] + lr[i]), None)
        if j is not None:
            nd[i] = float(lb[j] > lb[i])
        j = next((k for k in range(i + 1, len(d)) if bb[k] + br[k] > bb[i] + br[i]), None)
        if j is not None:
            nb[i] = float(bb[j] > bb[i])
    return pd.DataFrame(dict(game_id=gid, sec=E.sec(d._t), next_drake_blue=nd, next_baron_blue=nb))


def main():
    f = pd.read_parquet(f'{D}/v3/v2_logit.parquet')
    f['game_id'] = f.game_id.astype(str)
    f['sec'] = E.sec(f.ts)
    M = E.load_mm().drop_duplicates(['game_id', 'sec'])
    f = f[f.game_id.isin(set(M.game_id))].merge(M, on=['game_id', 'sec'], how='inner')
    ing = pd.read_parquet(f'{D}/ingame.parquet', columns=['game_id', 'ts', 'kills'])
    ing['game_id'] = ing.game_id.astype(str)
    ing['sec'] = E.sec(ing.ts.dt.tz_convert('UTC') if ing.ts.dt.tz is not None else ing.ts.dt.tz_localize('UTC'))
    f = f.merge(ing[['game_id', 'sec', 'kills']], on=['game_id', 'sec'], how='left')
    T = pd.concat([takers(g) for g in f.game_id.unique()])
    f = f.merge(T, on=['game_id', 'sec'], how='left')
    base = ['v2_logit', 'gold', 'alive', 'kills']
    rows = []
    for obj, up, lab, mmc, dc in [('drake/elder', (f.drake_up + f.elder_up) > 0, 'next_drake_blue', ['mm_drk', 'mm_wdrk', 'mm_adv', 'mm_vis'], ['d_drk', 'd1_drk', 'dd_miss', 'mm_wdrk', 'mm_adv']),
                                  ('baron', f.baron_up > 0, 'next_baron_blue', ['mm_bar', 'mm_wbar', 'mm_adv', 'mm_vis'], ['d_bar', 'd1_bar', 'dd_miss', 'mm_wbar', 'mm_adv'])]:
        g = f[up & f[lab].notna()].reset_index(drop=True)
        g['gold'] = g.gold / 1000
        y = g[lab].astype(int).to_numpy()
        folds = list(GroupKFold(5).split(g, groups=g.match_id))
        Q = {}
        for k, cols in (('feed', base), ('feed + minimap', base + mmc), ('feed + distances', base + dc), ('feed + both', base + list(dict.fromkeys(mmc + dc)))):
            q = np.zeros(len(g))
            for a, b in folds:
                q[b] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(g.loc[a, cols], y[a]).predict_proba(g.loc[b, cols])[:, 1]
            Q[k] = q
        pg = {k: pd.Series(ll(y, q), index=g.game_id).groupby(level=0).mean() for k, q in Q.items()}
        gm = g.drop_duplicates('game_id').set_index('game_id').match_id
        print(f'{obj}: frames {len(g)} games {g.game_id.nunique()} blue share {y.mean():.3f}')
        for k, q in Q.items():
            d, lo, hi = boot_diff(pg[k].to_numpy(), pg['feed'].to_numpy(), gm.reindex(pg['feed'].index).to_numpy())
            print(f'   {k:17s} log loss {ll(y, q).mean():.4f} AUC {roc_auc_score(y, q):.3f}  per-game vs feed {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        d, lo, hi = boot_diff(pg['feed + minimap'].to_numpy(), pg['feed'].to_numpy(), gm.reindex(pg['feed'].index).to_numpy())
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(g[base + dc], y)
        print('   distance model coefs per SD:', {k: round(v, 3) for k, v in zip(base + dc, m[-1].coef_[0])})
        rows.append(dict(obj=obj, n=len(g), ll_feed=ll(y, Q['feed']).mean(), ll_mm=ll(y, Q['feed + minimap']).mean(),
                         auc_feed=roc_auc_score(y, Q['feed']), auc_mm=roc_auc_score(y, Q['feed + minimap']), d=d, lo=lo, hi=hi))
    pd.DataFrame(rows).to_csv(f'{D}/../results/v3_taker.csv', index=False)


if __name__ == '__main__':
    main()
