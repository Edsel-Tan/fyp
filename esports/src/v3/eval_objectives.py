"""v3 step 3: do drake / baron / elder status (objectives.py) add to v2 on the full 60 s frame set?

v2 = eval_ingame2's `prior_new oof` (STATS + prior, each x (1, m, m^2)). The status flags are side-symmetric, so on their
own they cannot move a blue-minus-red logit; they enter through who is better placed to take the objective:
  {drake, baron, elder}_up x gold lead, x alive difference;  soul_pt;  soul_up (the next drake decides soul).
Also measured: the next-60 s |change| of v2's logit with each objective up vs not (the "volatility" the user expects).
Writes data/v3/obj_test_preds.parquet and results/v3_objectives.csv.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_pregame import BURN, SPLIT, ll, boot_diff
from eval_ingame import STATS, design
from eval_ingame2 import fitq

D = '../data'
OBJ = ['drake_up_gold', 'drake_up_alive', 'baron_up_gold', 'baron_up_alive', 'elder_up_gold', 'elder_up_alive',
       'soul_pt', 'soul_up']


def load():
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')].reset_index(drop=True)
    f = f.merge(pd.read_parquet(f'{D}/prior2.parquet')[['game_id', 'prior_new']], on='game_id', how='left')
    o = pd.read_parquet(f'{D}/v3/objectives.parquet')
    f = f.merge(o, on=['game_id', 'ts'], how='left')
    for k in ('drake', 'baron', 'elder'):
        f[f'{k}_up_gold'] = f[f'{k}_up'] * f.gold / 1000
        f[f'{k}_up_alive'] = f[f'{k}_up'] * f.alive
    return f


def main():
    f = load()
    miss = f.drake_up.isna()
    print(f'frames {len(f)}, without objective state {miss.sum()} (filled 0)')
    f[OBJ] = f[OBJ].fillna(0)
    f['prior_logit'] = f.prior_new
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    Q = {'v2': fitq(design(f, STATS + ['prior']), y, tr, te),
         'v2 + objectives': fitq(design(f, STATS + OBJ + ['prior']), y, tr, te)}
    ft = f[te].reset_index(drop=True)
    yt = y[te]
    gm = ft.drop_duplicates('game_id').set_index('game_id').match_id
    pg = {k: pd.Series(ll(yt, v), index=ft.game_id).groupby(level=0).mean() for k, v in Q.items()}
    d, lo, hi = boot_diff(pg['v2 + objectives'].to_numpy(), pg['v2'].to_numpy(), gm.reindex(pg['v2'].index).to_numpy())
    rows = [dict(model=k, per_game=pg[k].mean()) for k in Q] + [dict(model='diff', per_game=d, lo=lo, hi=hi)]
    print(f"v2 {pg['v2'].mean():.4f}  v2+obj {pg['v2 + objectives'].mean():.4f}  diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    # where the gain is: frames with an objective up
    for k in ('drake_up', 'baron_up', 'elder_up'):
        s = (ft[k] == 1).to_numpy()
        a, b = ll(yt[s], Q['v2 + objectives'][s]).mean(), ll(yt[s], Q['v2'][s]).mean()
        print(f'  {k}=1 frames {s.sum():6d}: v2 {b:.4f} v2+obj {a:.4f} ({a - b:+.4f})')
        rows.append(dict(model=f'{k} frames', per_game=a - b, n=int(s.sum())))
    # volatility: |logit change| over the next frame, v2, by objective state
    lg = np.log(Q['v2'] / (1 - Q['v2']))
    ft['lg'] = lg
    ft['dlg'] = (ft.groupby('game_id').lg.shift(-1) - ft.lg).abs()
    print('mean |next-60 s change in v2 logit|, by objective state (frames 15-35 min):')
    w = ft[(ft.clock > 900) & (ft.clock < 2100)]
    for k in ('drake_up', 'baron_up', 'elder_up', 'soul_up'):
        g = w.groupby(w[k].abs()).dlg.agg(['mean', 'median', 'size'])
        print(f'  {k}:', g.round(3).to_dict('index'))
    pd.DataFrame(rows).to_csv(f'{D}/../results/v3_objectives.csv', index=False)
    ft[['game_id', 'ts', 'clock']].assign(q_v2=Q['v2'], q_v2obj=Q['v2 + objectives']).to_parquet(f'{D}/v3/obj_test_preds.parquet')


if __name__ == '__main__':
    main()
