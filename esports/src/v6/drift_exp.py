"""2026 drift correction variants for v6 (inputs, intercept, window 3/6/12 months, x0.5 shrink), each refit monthly and causal.
sel = 2025-08..2026-01 test months, eval = 2026-02+, live = v5/live_period frames; per-game log loss minus v6 without drift.
Run after v6/build.py, from esports/src. Output: results/v6_drift.log."""
import os, sys, pickle
sys.path[:0]=['v6','v5','v4','.']
import numpy as np, pandas as pd, statsmodels.api as sm
import build as V6
from eval_pregame import SPLIT, ll
pv=pd.read_parquet('../data/ingame_test_preds_v6.parquet')
import importlib; B5=importlib.import_module('build')
f=pd.read_parquet('../data/ingame.parquet', columns=['game_id','ts','gold','kills','towers','alive'])
pv=pv.drop(columns=['gold','kills','towers','alive']).merge(f,on=['game_id','ts'],how='left') if 'gold' not in pv else pv
pv['zp']=V6.lg(pv.q_v6_pers.to_numpy())
F=pd.read_parquet('../data/v6/live_period_preds.parquet'); F['zp']=V6.lg(F.q_v6_pers.to_numpy())
H=pd.concat([pv[['game_id','t_start','clock','gold','kills','towers','alive','blue_win','zp']], F[~F.game_id.isin(pv.game_id)][['game_id','t_start','clock','gold','kills','towers','alive','blue_win','zp']]])
def X(d, cols, icpt, inter):
    m=d.clock.to_numpy()/600; z=d.assign(gold_k=d.gold/1000)[cols].astype(float).to_numpy()
    parts=([np.ones(len(d))[:,None]] if icpt else [])+[z]+([z*m[:,None]] if inter else [])
    return np.hstack(parts)
def run(T, cols, icpt, inter, months, shrink):
    z=T.zp.to_numpy().copy()
    for m0 in pd.date_range('2025-08-01', '2026-10-01', freq='MS', tz='UTC'):
        cur=((T.t_start>=m0)&(T.t_start<m0+pd.offsets.MonthBegin(1))).to_numpy()
        w=H[(H.t_start>=m0-pd.DateOffset(months=months))&(H.t_start<m0)]
        if not cur.any() or w.game_id.nunique()<300: continue
        p=sm.GLM(w.blue_win.astype(int).to_numpy(), X(w,cols,icpt,inter), family=sm.families.Binomial(), offset=w.zp.to_numpy()).fit().params
        z[cur]+=shrink*(X(T[cur],cols,icpt,inter)@p)
    return z
def pg(T,z): return pd.Series(ll(T.blue_win.astype(int).to_numpy(), 1/(1+np.exp(-z))), index=T.game_id).groupby(level=0).mean()
S=pd.Timestamp('2026-02-01',tz='UTC')
for cols in (['gold_k'], ['gold_k','kills'], V6.DRIFT):
  for icpt in (False, True):
    for months in (3, 6, 12):
      for shrink in (1.0, 0.5):
        zt=run(pv, cols, icpt, True, months, shrink); zl=run(F, cols, icpt, True, months, shrink)
        b=pg(pv,pv.zp.to_numpy()); a=pg(pv,zt); gt=pv.drop_duplicates('game_id').set_index('game_id').t_start
        sel=(gt<S)&(gt>=pd.Timestamp('2025-08-01',tz='UTC')); ev=gt>=S
        bl, al = pg(F,F.zp.to_numpy()), pg(F,zl)
        print(f'{"+".join(cols):28s} icpt {icpt!s:5s} {months:2d}m x{shrink}: sel {a[sel.reindex(a.index).values].mean()-b[sel.reindex(b.index).values].mean():+.4f}  eval {a[ev.reindex(a.index).values].mean()-b[ev.reindex(b.index).values].mean():+.4f}  live {al.mean()-bl.mean():+.4f}', flush=True)
