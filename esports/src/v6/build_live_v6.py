"""Freeze v6 for the live tracker (~/fyp-pmlive/lollive, src/v6live.py) and check its prior code path.

Writes ~/fyp-pmlive/lollive/data/v6/:
  v6_model.pkl       v5's in-game logistic (model, cols, xcols), the union prior's C, the pre-game blend (w0, w1) from
                     v6/build.py, and v6m's per-phase weights per latency from v6/book_sim_v6.py (data/v6/comb.pkl)
  e_gold.parquet     end gold difference of the study's games (the Kalman update's margin of victory)
  dmg.parquet        end damage shares of the study's games (reddit/build's cache)
  last_state.parquet the same for games after the study's table (v6/prior.py's cache; the tracker extends it)
  priors_study.parquet  v6/prior.py's causal priors (backfill_v6.py's z_u for games logged before v6 went live)
Check (run with lollive's interpreter, as the trader does): v6live.features on the whole table and the October refit, against
v6/prior.py's causal prior_union_m for every October game (2026-10-13: max difference 5.6e-17 over 202 games).
Usage: python v6/build_live_v6.py [check]
"""
import os, sys, pickle, shutil, subprocess, json
import numpy as np, pandas as pd

D = '../data'
LL = os.path.expanduser('~/fyp-pmlive/lollive')
OUT = f'{LL}/data/v6'


def build():
    os.makedirs(OUT, exist_ok=True)
    V = pickle.load(open(f'{D}/v6/models.pkl', 'rb'))
    C = pickle.load(open(f'{D}/v6/comb.pkl', 'rb'))['live']  # archive weights (v6/book_sim_v6.py docstring)
    M5 = V['v5']
    assert V['xprior'] == 'prior_union_m' and not np.any(V['persist']) and V['drift'] is None
    pickle.dump(dict(v5=dict(model=M5['model'], cols=M5['cols'], xcols=M5['xcols']), prior_C=0.01, blend=[float(x) for x in V['blend']],
                     comb=C, built=str(pd.Timestamp.now(tz='UTC')), src='~/fyp/esports/src/v6/build_live_v6.py'),
                open(f'{OUT}/v6_model.pkl', 'wb'))
    pd.read_parquet(f'{D}/game_table.parquet', columns=['game_id', 'e_gold']).to_parquet(f'{OUT}/e_gold.parquet')
    shutil.copy(f'{D}/reddit/dmg.parquet', f'{OUT}/dmg.parquet')
    shutil.copy(f'{D}/v6/priors.parquet', f'{OUT}/priors_study.parquet')  # for lollive src/backfill_v6.py
    if not os.path.exists(f'{OUT}/last_state.parquet'):
        shutil.copy(f'{D}/v6/last_state.parquet', f'{OUT}/last_state.parquet')
    print(f'wrote {OUT}: blend {V["blend"]}, comb latencies {sorted(C["comb"])}')


CHECK = r'''
import sys, json, pandas as pd
sys.path.insert(0, 'src')
import v6live
V = v6live.V6()
P = v6live.features([])
m0 = pd.Timestamp('2026-10-01', tz='UTC')
a = (P.t_start >= v6live.BURN) & (P.t_end < m0) & P.blue_win.notna()
X = P.loc[a, v6live.UNION]
sc = X.std().replace(0, 1)
from sklearn.linear_model import LogisticRegression
m = LogisticRegression(C=V.M['prior_C'], max_iter=5000).fit(X / sc, P.loc[a, 'blue_win'].astype(int))
x = P[P.t_start >= m0]
print(json.dumps(dict(n_fit=int(a.sum()), z=dict(zip(x.game_id, m.decision_function(x[v6live.UNION] / sc).tolist())),
                      cross=dict(zip(x.game_id, x.cross.tolist())))))
'''


def check():
    """The tracker's feature pass on the whole table (no live rows) and its October refit, against v6/prior.py's prior_union_m.
    (Treating finished games as live rows is not a clean check: hiding their outcomes changes later games' ratings.)"""
    PR = pd.read_parquet(f'{D}/v6/priors.parquet').set_index('game_id')
    r = subprocess.run([f'{LL}/../.venv/bin/python', '-c', CHECK], cwd=LL, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit(1)
    got = json.loads(r.stdout.strip().splitlines()[-1])
    z = pd.Series(got['z'])
    d = (z - PR.prior_union_m.reindex(z.index)).abs()
    cr = (pd.Series(got['cross']) == PR.cross.reindex(z.index)).mean()
    print(f'October refit on {got["n_fit"]} games; {len(z)} October games: max |tracker - study prior_union_m| {d.max():.2e}; cross agrees {cr:.0%}')


if __name__ == '__main__':
    build()
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        check()
