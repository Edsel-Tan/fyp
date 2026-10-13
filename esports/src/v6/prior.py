"""v6 pre-game features and priors on the extended game table (the study's table + lollive's games_new).

The PROGRESS "Why the market beats the models live" list, item 1 (cross-league team strength), is addressed in two places:
  (a) here: the union prior (v5) plus cross-league terms, i.e. `cross` (the two teams' home leagues differ) and its
      interaction with each rating family, so the prior can reweight which rating to trust when ERLs, academies or
      regions meet (league_elo only updates on rare inter-league games);
  (b) v6/build.py: the market's pre-game price blended into the prior (weights fit on the backtest's selection months).

Everything is causal: pregame.build / pregame2.build / reddit.build's passes on the extended table (v5/live_period.py's
code), then standardised logistics refit at each month start on every game finished before it.
  prior_union_m   v5's prior (reproduces the study's prior_both_m on its test games; checked by live_period.priors)
  prior_x_m       the union set + XCOLS (C chosen on the validation slice, the last 91 days of train)
  pm_pre          the market's pre-game P(blue): the Polymarket-v1 archive's last-30-min VWAP for study games
                  (v4/pregame_f), else the captured book's median mid over the 10 min before the first frame (gap/panel)
End-of-game states of the new games (gold difference, damage shares; reddit's lane / damage ratings) are cached in
data/v6/last_state.parquet, so a rebuild only fetches games it has not seen.
Usage: python v6/prior.py   (writes data/v6/prior_feats.parquet, data/v6/priors.parquet; prints the validation choice
and test log loss)
"""
import os, sys, time, json
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5'), os.path.join(SRC, 'reddit')]
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff
import live_period as LP

D = '../data'
LLD = os.path.expanduser('~/fyp-pmlive/lollive/data')
CACHE = f'{D}/v6/last_state.parquet'
VAL0 = SPLIT - pd.Timedelta('91D')
XBASE = ['elo_team_32', 'elo_play_32', 'league_elo', 'bt', 'kal_z', 'form_10']


def _frames_path(gid):
    for p in (f'{D}/{LP.NEWF}/{gid}.parquet', f'{LLD}/frames60/{gid}.parquet'):
        if os.path.exists(p):
            return p
    return None


def last_state(gid):
    """LP.last_state on whichever saved feed exists (v5's 60 s download, else the tracker's own dense feed)."""
    import lolapi
    from collect_frames import iso, pt
    out = dict(game_id=gid, e_gold=np.nan)
    fp = _frames_path(gid)
    if fp is None:
        return out
    d = pd.read_parquet(fp)
    d = d[(d.b_gold + d.r_gold) > 0]
    if not len(d):
        return out
    out['e_gold'] = float(d.b_gold.iloc[-1] - d.r_gold.iloc[-1])
    for t in list(d.ts)[::-1][:4]:
        x = lolapi.details(gid, iso(pt(t)))
        if x and x.get('frames'):
            ps = {p['participantId']: p.get('championDamageShare') for p in x['frames'][0]['participants']}
            if all(ps.get(i) is not None for i in range(1, 11)) and sum(ps.values()) > 1.5:
                out.update({f'dmg{i}': float(ps[i]) for i in range(1, 11)})
                break
    return out


def last_states(gids):
    c = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pd.DataFrame(columns=['game_id', 'e_gold'])
    todo = [g for g in gids if g not in set(c.game_id[c.e_gold.notna()])]
    if todo:
        with ThreadPoolExecutor(16) as ex:
            new = pd.DataFrame(list(ex.map(last_state, todo)))
        c = pd.concat([c[~c.game_id.isin(new.game_id)], new], ignore_index=True)
        c.to_parquet(CACHE)
    return c[c.game_id.isin(gids)].reset_index(drop=True)


def reddit_feats(p, rg):
    """LP.reddit_feats with the cached end states. Games with no end state (a live row) keep e_gold / damage NaN,
    which reddit's passes treat as unobserved (the game's own outcome is never used for its own features)."""
    import build as R
    q = p.copy()
    q['intl'] = q.league.isin(INTL)
    new = q.src.eq('lollive') & q.blue_win.notna()
    ls = last_states(list(q.game_id[new]))
    dm = pd.read_parquet(f'{D}/reddit/dmg.parquet')
    dcols = [f'dmg{i}' for i in range(1, 11)]
    dm = pd.concat([dm, ls[['game_id'] + [c for c in dcols if c in ls]].dropna()], ignore_index=True).drop_duplicates('game_id')
    q = q.merge(ls[['game_id', 'e_gold']].rename(columns={'e_gold': 'e_gold_new'}), on='game_id', how='left')
    q['e_gold'] = q.e_gold.fillna(q.e_gold_new)
    q = q.merge(rg[['game_id'] + [f'gd15_{k}' for k in range(5)]], on='game_id', how='left').merge(dm, on='game_id', how='left')
    q = q.sort_values('t_start').reset_index(drop=True)
    ch = R.Champs(q)
    dr = R.draft_models(q, ch)
    f = R.player_pass(q, dr, R.KAL)
    f['bt'] = R.bt_ratings(q)
    f['draft_wr'] = dr.draft_wr.to_numpy()
    f['draft_gd'] = sum(dr[f'e_gd_{k}'] for k in range(5)).to_numpy() / 1000
    return f


def features(e=None, verbose=True):
    """Pre-game feature table for every game of e (default: LP.ext_table()). Rows of e with blue_win NaN (live games)
    get features too; they never enter anyone else's state, since they have no end."""
    t0 = time.time()
    e = LP.ext_table() if e is None else e
    p, f2, rg = LP.pregame_all(e)
    t1 = time.time()
    fr = reddit_feats(p, rg)
    P = p.merge(f2, on='game_id').merge(fr, on='game_id')
    P['cross'] = (P.b_league != P.r_league).astype(float)
    P['intl'] = P.league.isin(INTL).astype(float)
    if verbose:
        print(f'features: {len(P)} games; pregame passes {t1 - t0:.0f} s, reddit passes {time.time() - t1:.0f} s', flush=True)
    return P


def xcols(P):
    for c in XBASE:
        P[f'x_{c}'] = P[c] * P.cross
    return ['cross'] + [f'x_{c}' for c in XBASE]


def fitpred(P, cols, C, a, y):
    sc = P.loc[a, cols].std().replace(0, 1)
    m = LogisticRegression(C=C, max_iter=5000).fit(P.loc[a, cols] / sc, y[a])
    return m.decision_function(P[cols] / sc), (m, sc)


def monthly(P, cols, C, start=SPLIT):
    """Refit at each month start on every finished game since BURN; returns the logit for games from `start` on."""
    y = P.blue_win.fillna(0).astype(int).to_numpy()
    out = pd.Series(np.nan, index=P.index)
    for m0 in pd.date_range(start, P.t_start.max() + pd.offsets.MonthBegin(1), freq='MS'):
        cur = ((P.t_start >= m0) & (P.t_start < m0 + pd.offsets.MonthBegin(1))).to_numpy()
        if cur.any():
            fit = ((P.t_end < m0) & P.blue_win.notna()).to_numpy()
            out[cur] = fitpred(P, cols, C, fit, y)[0][cur]
    return out


def pm_pre_all():
    a = pd.read_parquet(f'{D}/v4/pregame_f.parquet')[['game_id', 'pm_pre']].dropna()
    b = pd.read_parquet(f'{D}/gap/games.parquet')[['game_id', 'pm_pre']].dropna()
    x = pd.concat([a, b[~b.game_id.isin(a.game_id)]], ignore_index=True)
    return x[(x.pm_pre > 0) & (x.pm_pre < 1)]


def main():
    import eval as RE
    P = features()
    P.to_parquet(f'{D}/v6/prior_feats.parquet')
    P = LP.priors(P)  # prior_new, prior_union, prior_union_m (+ the reproduction check)
    UN = RE.V4P + RE.POST
    XC = xcols(P)
    y = P.blue_win.fillna(0).astype(int).to_numpy()
    fit, va = (P.t_end < VAL0).to_numpy(), ((P.t_start >= VAL0) & (P.t_start < SPLIT)).to_numpy()
    vc = P.cross.to_numpy()[va] == 1
    print(f'validation slice {va.sum()} games ({vc.sum()} cross-league)')
    res = {}
    for nm, cols in (('union', UN), ('union + cross', UN + XC)):
        for C in (0.003, 0.01, 0.03, 0.1):
            z = fitpred(P, cols, C, fit, y)[0][va]
            L = ll(y[va], 1 / (1 + np.exp(-z)))
            res[(nm, C)] = L.mean()
            print(f'  {nm:14s} C {C:<6} val LL {L.mean():.4f}  cross {L[vc].mean():.4f}  same {L[~vc].mean():.4f}')
    Cx = min((C for (n, C) in res if n == 'union + cross'), key=lambda C: res[('union + cross', C)])
    print(f'chosen C for union + cross: {Cx}')
    P['prior_x_m'] = monthly(P, UN + XC, Cx)
    P = P.merge(pm_pre_all(), on='game_id', how='left')
    keep = ['game_id', 'match_id', 't_start', 't_end', 'league', 'b_league', 'r_league', 'cross', 'intl', 'src', 'blue_win',
            'n_min_games', 'prior_new', 'prior_union', 'prior_union_m', 'prior_x_m', 'pm_pre']
    P[keep].to_parquet(f'{D}/v6/priors.parquet')
    te = (P.t_start >= SPLIT) & P.blue_win.notna()
    T = P[te]
    yt = T.blue_win.astype(int).to_numpy()
    a, b = ll(yt, 1 / (1 + np.exp(-T.prior_x_m))), ll(yt, 1 / (1 + np.exp(-T.prior_union_m)))
    d = boot_diff(a, b, T.match_id.to_numpy())
    print(f'test ({len(T)} games): prior_x_m {a.mean():.4f}  prior_union_m {b.mean():.4f}  diff {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]')
    for k, m in (('cross', T.cross.to_numpy() == 1), ('same', T.cross.to_numpy() == 0)):
        d = boot_diff(a[m], b[m], T.match_id.to_numpy()[m])
        print(f'  {k}-league {m.sum()}: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]')


if __name__ == '__main__':
    main()
