"""Draft and series study, step 1: monthly-refit pre-game priors with and without the draft inputs (2026-10-16).

Every prior family the models use is refit at each month start on every game finished before it (v6/prior.py's protocol),
once with its full feature set and once with the draft-dependent columns removed:
  v2   prior_new's set (eval_pregame2 'all new minus h2h'; the prior of v2 and v4)          C 0.03
  rd   the reddit model's post-draft stack (reddit/eval POST)                                C 0.01
  un   the union set (v2 + reddit POST; the prior of v5 / v6)                                C 0.01
Draft columns: champ_all, champ_patch, lane_mu, pc_exp, pc_wr (v2) and draft_wr, draft_gd (reddit).
Pre-draft = same set minus those, same C. The difference z_full - z_pre is "what the draft did" to each model's prior.
Check: un_full reproduces v6/priors.parquet prior_union_m.
Writes data/draftseries/priors.parquet (game rows from SPLIT on, plus the train rows with an out-of-fold-in-time NaN).
Usage (from esports/src): python draftseries/priors.py
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'reddit'), os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5'), os.path.join(SRC, 'v6')]
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff
import eval as RE

D = os.path.join(SRC, '../data')
DRAFT = ['champ_all', 'champ_patch', 'lane_mu', 'pc_exp', 'pc_wr', 'draft_wr', 'draft_gd']
SETS = {'v2': (RE.V4P, 0.03), 'rd': (RE.POST, 0.01), 'un': (RE.V4P + RE.POST, 0.01)}


def load():
    P = pd.read_parquet(f'{D}/v6/prior_feats.parquet')
    P = P[P.t_start >= BURN].sort_values('t_start').reset_index(drop=True)
    P['age_x_elo'] = P.elo_play_32 * np.exp(-P.patch_age / 7)
    return P


def monthly(P, cols, C, start=SPLIT, extra=None):
    """Logit refit at each month start on games finished before it. `extra`: callable(P, fit_mask, cur_mask) -> (Xfit, Xcur)
    of extra columns built per refit (unused here, kept for the draft candidates)."""
    y = P.blue_win.fillna(0).astype(int).to_numpy()
    out = np.full(len(P), np.nan)
    for m0 in pd.date_range(start, P.t_start.max() + pd.offsets.MonthBegin(1), freq='MS'):
        cur = ((P.t_start >= m0) & (P.t_start < m0 + pd.offsets.MonthBegin(1))).to_numpy()
        if not cur.any():
            continue
        fit = ((P.t_end < m0) & P.blue_win.notna()).to_numpy()
        X = P[cols]
        sc = X[fit].std().replace(0, 1)
        m = LogisticRegression(C=C, max_iter=5000).fit(X[fit] / sc, y[fit])
        out[cur] = m.decision_function(X[cur] / sc)
    return out


def main():
    P = load()
    out = P[['game_id', 'match_id', 'number', 'bo', 't_start', 't_end', 'league', 'b_league', 'r_league', 'cross', 'intl',
             'blue_id', 'red_id', 'blue_win', 'src', 'series_diff', 'n_min_games', 'e_gold', 'dur_s', 'patch']].copy()
    for k, (cols, C) in SETS.items():
        pre = [c for c in cols if c not in DRAFT]
        out[f'{k}_full'] = monthly(P, cols, C)
        out[f'{k}_pre'] = monthly(P, pre, C)
        print(f'{k}: {len(cols)} cols, {len(cols) - len(pre)} draft', flush=True)
    chk = out.merge(pd.read_parquet(f'{D}/v6/priors.parquet')[['game_id', 'prior_union_m']], on='game_id')
    chk = chk[chk.prior_union_m.notna()]
    print(f'check un_full vs v6 prior_union_m on {len(chk)} games: max |diff| {np.abs(chk.un_full - chk.prior_union_m).max():.2e}')
    out.to_parquet(f'{D}/draftseries/priors.parquet')
    T = out[(out.t_start >= SPLIT) & out.blue_win.notna()]
    y = T.blue_win.astype(int).to_numpy()
    for k in SETS:
        a, b = ll(y, 1 / (1 + np.exp(-T[f'{k}_full']))), ll(y, 1 / (1 + np.exp(-T[f'{k}_pre'])))
        d = boot_diff(a, b, T.match_id.to_numpy())
        print(f'{k}: full {a.mean():.4f} pre {b.mean():.4f}  full - pre {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]')


if __name__ == '__main__':
    main()
