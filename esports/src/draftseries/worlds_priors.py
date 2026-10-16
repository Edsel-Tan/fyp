"""MVK-FUR / Worlds case: the tracker's v6 union prior (October refit) with and without the draft columns, plus the
rating features, for every Worlds game so far. Run with lollive's interpreter from lollive/src:
  cd ~/fyp-pmlive/lollive/src && ../../.venv/bin/python ~/fyp/esports/src/draftseries/worlds_priors.py
Writes ~/fyp/esports/data/draftseries/worlds_priors.parquet.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/fyp-pmlive/lollive/src'))
import v6live
from common import BURN
from sklearn.linear_model import LogisticRegression

DRAFT = ['champ_all', 'champ_patch', 'lane_mu', 'pc_exp', 'pc_wr', 'draft_wr', 'draft_gd']
P = v6live.features([])
m0 = pd.Timestamp('2026-10-01', tz='UTC')
fit = (P.t_start >= BURN) & (P.t_end < m0) & P.blue_win.notna()
W = P[P.league.eq('worlds')].copy()
for name, cols in (('full', v6live.UNION), ('pre', [c for c in v6live.UNION if c not in DRAFT])):
    sc = P.loc[fit, cols].std().replace(0, 1)
    m = LogisticRegression(C=0.01, max_iter=5000).fit(P.loc[fit, cols] / sc, P.loc[fit, 'blue_win'].astype(int))
    W[f'z_{name}'] = m.decision_function(W[cols] / sc)
    if name == 'full':
        co = pd.Series(m.coef_[0], index=cols)
        contrib = (W[cols] / sc) * co
        for c in ('elo_team_32', 'elo_play_32', 'kal_z', 'bt', 'league_elo', 'series_diff', 'form_5', 'form_10', 'h2h'):
            W[f'c_{c}'] = contrib[c]
        W['c_draft'] = contrib[DRAFT].sum(axis=1)
keep = ['game_id', 'match_id', 'number', 't_start', 'blue_id', 'red_id', 'blue_win', 'cross', 'z_full', 'z_pre', 'series_diff',
        'elo_team_32', 'kal_z', 'bt', 'league_elo'] + [c for c in W.columns if c.startswith('c_')]
W[keep].to_parquet(os.path.expanduser('~/fyp/esports/data/draftseries/worlds_priors.parquet'))
pd.set_option('display.width', 250)
print(W[keep].drop(columns=['blue_id', 'red_id', 'match_id']).round(3).to_string())
