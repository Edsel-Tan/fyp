"""Freeze v7 for the live tracker (~/fyp-pmlive/lollive, src/v7live.py) and check its prior code path.

Writes ~/fyp-pmlive/lollive/data/v7/:
  v7_model.pkl        v5's in-game logistic, the v7 prior's columns / C / Kalman settings (data/v7/choice.json), the pre-game
                      blend (w0, w1) from v7/build.py, and v7m's per-phase weights per latency (v7/backtest_v7.py's archive fit)
  g_area.parquet      per-game dominance (mean tanh(gold diff / 5k)) of every game in the study's extended table
  priors_study.parquet  v7/prior.py's causal priors (backfill_v7.py's z for games logged before v7 went live)
and copies src/v7/v7feat.py to lollive/src/v7feat.py (verbatim).
Check (lollive's interpreter): v7live.features on the whole table and the October refit vs v7/prior.py's prior_v7_m.
Usage: python v7/build_live_v7.py [check]
"""
import os, sys, pickle, shutil, subprocess, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '../../data')
LL = os.path.expanduser('~/fyp-pmlive/lollive')
OUT = f'{LL}/data/v7'


def build():
    os.makedirs(OUT, exist_ok=True)
    V = pickle.load(open(f'{D}/v7/models.pkl', 'rb'))
    C = pickle.load(open(f'{D}/v7/comb.pkl', 'rb'))['live']
    ch = json.load(open(f'{D}/v7/choice.json'))
    M5 = V['v5']
    assert not np.any(V['persist'])
    pickle.dump(dict(v5=dict(model=M5['model'], cols=M5['cols'], xcols=M5['xcols']), prior_C=ch['C'], cols=ch['cols'], kalman=ch['kalman'],
                     blend=[float(x) for x in V['blend']], comb=C, built=str(pd.Timestamp.now(tz='UTC')), src='~/fyp/esports/src/v7/build_live_v7.py'),
                open(f'{OUT}/v7_model.pkl', 'wb'))
    pd.read_parquet(f'{D}/v7/feat.parquet', columns=['game_id', 'g_area']).dropna().to_parquet(f'{OUT}/g_area.parquet')
    shutil.copy(f'{D}/v7/priors.parquet', f'{OUT}/priors_study.parquet')
    shutil.copy(os.path.join(HERE, 'v7feat.py'), f'{LL}/src/v7feat.py')
    print(f'wrote {OUT}: blend {V["blend"]}, {len(ch["cols"])} prior cols, kalman {ch["kalman"]}, comb latencies {sorted(C["comb"])}')


CHECK = r'''
import sys, json, pandas as pd
sys.path.insert(0, 'src')
import v7live
from sklearn.linear_model import LogisticRegression
V = v7live.V7()
P = v7live.features([], V.M['kalman'])
m0 = pd.Timestamp('2026-10-01', tz='UTC')
a = (P.t_start >= v7live.BURN) & (P.t_end < m0) & P.blue_win.notna()
cols = V.M['cols']
X = P.loc[a, cols]
sc = X.std().replace(0, 1)
m = LogisticRegression(C=V.M['prior_C'], max_iter=5000).fit(X / sc, P.loc[a, 'blue_win'].astype(int))
x = P[P.t_start >= m0]
print(json.dumps(dict(n_fit=int(a.sum()), z=dict(zip(x.game_id, m.decision_function(x[cols] / sc).tolist())))))
'''


def check():
    PR = pd.read_parquet(f'{D}/v7/priors.parquet').set_index('game_id')
    r = subprocess.run([f'{LL}/../.venv/bin/python', '-c', CHECK], cwd=LL, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit(1)
    got = json.loads(r.stdout.strip().splitlines()[-1])
    z = pd.Series(got['z'])
    both = z.index[z.index.isin(PR.index)]
    d = (z[both] - PR.prior_v7_m.reindex(both)).abs()
    print(f'October refit on {got["n_fit"]} games; {len(both)} October games in the study table: max |tracker - study prior_v7_m| '
          f'{d.max():.2e}, median {d.median():.2e}; {len(z) - len(both)} newer games')


if __name__ == '__main__':
    build()
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        check()
