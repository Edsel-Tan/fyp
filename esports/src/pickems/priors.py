"""v6 pre-game priors for every pairing of the Worlds 2026 field (pick'ems, 2026-10-15).

Runs the live tracker's v6 feature pass (~/fyp-pmlive/lollive/src/v6live.features: pregame + pregame2 + the reddit rating
passes on the whole table) with one hypothetical live row per ordered team pair, each team on its latest roster.
Pick'ems are made before the draft, so two union logistics are fit (v6's protocol: every game finished before the
month start, C = 0.01, standardised):
  full      v6's union prior (draft terms included; the live rows get their last game's champions as placeholders)
  predraft  the same set minus every draft-dependent column (DRAFT below)
Both are scored out of sample on October's finished games. Side is unknown, so P(A beats B in a game) averages
A on blue and A on red.
Run with lollive's interpreter from lollive/src:
  cd ~/fyp-pmlive/lollive/src && ../../.venv/bin/python ~/fyp/esports/src/pickems/priors.py
Writes ~/fyp/esports/data/pickems/priors.parquet (one row per ordered pair) and teams.json.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser('~/fyp-pmlive/lollive/src'))
import state, v6live
from common import BURN
from sklearn.linear_model import LogisticRegression

OUT = os.path.expanduser('~/fyp/esports/data/pickems')
DOMESTIC_EXCL = {'wsci', 'lck_challengers_league', 'cd'}  # DK's latest game is WSCI with its academy roster
TEAMS = {  # code: (esportsTeamId, region)
    'GEN': ('100205573495116443', 'LCK'), 'HLE': ('100205573496804586', 'LCK'), 'T1': ('98767991853197861', 'LCK'),
    'DK': ('100725845018863243', 'LCK'),
    'AL': ('99566404856367466', 'LPL'), 'BLG': ('99566404853854212', 'LPL'), 'TES': ('99566404854685458', 'LPL'),
    'IG': ('99566404848691211', 'LPL'),
    'G2': ('98767991926151025', 'LEC'), 'MKOI': ('103461966965149786', 'LEC'), 'KC': ('111692118851466302', 'LEC'),
    'TLAW': ('98926509885559666', 'LCS'), 'LYON': ('99566405941863385', 'LCS'), 'C9': ('98767991877340524', 'LCS'),
    'TSW': ('113661839307879869', 'LCP'), 'CFO': ('107700199633958891', 'LCP'), 'MVK': ('107251245690956393', 'LCP'),
    'LOS': ('109480204628225868', 'CBLOL'), 'FUR': ('100205576309502431', 'CBLOL'),
}
DRAFT = ['champ_all', 'champ_patch', 'lane_mu', 'pc_exp', 'pc_wr', 'draft_wr', 'draft_gd']
T0 = pd.Timestamp('2026-10-15 18:00', tz='UTC')


def lineups(t):
    out = {}
    for c, (tid, _) in TEAMS.items():
        g = t[((t.blue_id == tid) | (t.red_id == tid)) & ~t.league.isin(DOMESTIC_EXCL)].sort_values('t_start')
        l = g.iloc[-1]
        s = 'b' if l.blue_id == tid else 'r'
        out[c] = dict(players=list(l[f'{s}_players']), champs=list(l[f'{s}_champs']), roles=list(l[f'{s}_roles']),
                      last=str(l.t_start)[:10], league=l.league, n=len(g))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t = state.table()
    t['t_start'] = pd.to_datetime(t.t_start, utc=True)
    L = lineups(t)
    patch = t.sort_values('t_start').patch.dropna().iloc[-1]
    rows = []
    for a in TEAMS:
        for b in TEAMS:
            if a == b:
                continue
            rows.append(dict(game_id=f'pk-{a}-{b}', blue_id=TEAMS[a][0], red_id=TEAMS[b][0], patch=patch,
                             b_players=L[a]['players'], b_champs=L[a]['champs'], b_roles=L[a]['roles'],
                             r_players=L[b]['players'], r_champs=L[b]['champs'], r_roles=L[b]['roles'],
                             t_start=T0, match_id=f'pk-{a}-{b}', number=1, league='worlds'))
    P = v6live.features(rows)
    m0 = pd.Timestamp('2026-10-01', tz='UTC')
    fit = (P.t_start >= BURN) & (P.t_end < m0) & P.src.ne('live') & P.blue_win.notna()
    oct_ = (P.t_start >= m0) & P.src.ne('live') & P.blue_win.notna()
    live = P[P.src.eq('live')].copy()
    res = {}
    for name, cols in (('full', v6live.UNION), ('predraft', [c for c in v6live.UNION if c not in DRAFT])):
        X = P.loc[fit, cols]
        sc = X.std().replace(0, 1)
        m = LogisticRegression(C=0.01, max_iter=5000).fit(X / sc, P.loc[fit, 'blue_win'].astype(int))
        y = P.loc[oct_, 'blue_win'].astype(int)
        q = np.clip(m.predict_proba(P.loc[oct_, cols] / sc)[:, 1], 1e-6, 1 - 1e-6)
        res[name] = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        live[f'z_{name}'] = m.decision_function(live[cols] / sc)
    print(f'fit on {int(fit.sum())} games; October out-of-sample LL ({int(oct_.sum())} games): {res}', flush=True)
    live['a'] = live.game_id.str.split('-').str[1]
    live['b'] = live.game_id.str.split('-').str[2]
    live['cross'] = live.cross
    keep = ['a', 'b', 'cross', 'z_full', 'z_predraft']
    live[keep].to_parquet(f'{OUT}/priors.parquet')
    json.dump(dict(teams={c: dict(id=v[0], region=v[1], **L[c]) for c, v in TEAMS.items()}, patch=patch, oct_ll=res,
                   n_fit=int(fit.sum())), open(f'{OUT}/teams.json', 'w'), indent=1, default=str)
    print(f'wrote {OUT}/priors.parquet ({len(live)} pairs)')


if __name__ == '__main__':
    main()
