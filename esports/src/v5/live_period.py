"""Score v2 / v4 / v5 on every game from START (the Polymarket book capture's first day) to the end of lollive's game list.

The study's tables end 2026-09-28; lollive's daily refresh (games_new.parquet, labelled with the frozen winner rule) continues them.
Everything is recomputed causally on the extended table, with the frozen models:
  1. pregame.build + pregame2.build (lollive's state.py does the same per live game), and reddit/build's ratings
     (e_gold and the end damage share of the new games come from their 60 s frames and one `details` request)
  2. priors: prior_new (v2/v4: the frozen train fit of eval_pregame2's set) and the union prior (v5), fixed train fit and
     monthly refit (on every game finished before the month start); a check reproduces the study's test priors
  3. in-game features on 60 s frames (data/frames60 for study games, data/v5/frames60 for new ones; ingame_data.one +
     v4/features.window_feats), then the frozen v2 / v4 (`sel window`) / v5 models
Writes data/v5/live_period_preds.parquet (one row per 60 s frame) and data/v5/live_period_games.parquet.
"""
import os, sys, pickle, math, json
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC); sys.path.insert(0, os.path.join(SRC, 'v4')); sys.path.insert(0, os.path.join(SRC, 'reddit'))
from eval_pregame import BURN, SPLIT, INTL, ll
import pregame, pregame2

D = '../data'
LL = os.path.expanduser('~/fyp-pmlive/lollive')
START = pd.Timestamp('2026-09-17', tz='UTC')
NEWF = 'v5/frames60'


def ext_table():
    t = pd.read_parquet(f'{D}/game_table.parquet')
    n = pd.read_parquet(f'{LL}/data/games_new.parquet')
    n = n[~n.game_id.isin(t.game_id)].copy()
    n['src'] = 'lollive'
    t['src'] = 'study'
    e = pd.concat([t, n], ignore_index=True)
    e = e[e.b_players.map(lambda l: all(l)) & e.r_players.map(lambda l: all(l))]
    return e.sort_values('t_start').reset_index(drop=True)


def last_state(gid):
    """End gold difference and end damage share for a new game (60 s frames + one details request)."""
    import lolapi
    from collect_frames import iso, pt
    d = pd.read_parquet(f'{D}/{NEWF}/{gid}.parquet')
    d = d[(d.b_gold + d.r_gold) > 0]
    out = dict(game_id=gid, e_gold=float(d.b_gold.iloc[-1] - d.r_gold.iloc[-1]) if len(d) else np.nan)
    for t in list(d.ts)[::-1][:4]:
        x = lolapi.details(gid, iso(pt(t)))
        if x and x.get('frames'):
            ps = {p['participantId']: p.get('championDamageShare') for p in x['frames'][0]['participants']}
            if all(ps.get(i) is not None for i in range(1, 11)) and sum(ps.values()) > 1.5:
                out.update({f'dmg{i}': float(ps[i]) for i in range(1, 11)})
                break
    return out


def pregame_all(e):
    p = pregame.build(e.copy())
    p['pv'] = p.patch.str.split('.').str[:2].str.join('.')
    tl = pd.concat([p[['blue_id', 'league']].rename(columns={'blue_id': 'tid'}), p[['red_id', 'league']].rename(columns={'red_id': 'tid'})])
    home = tl[~tl.league.isin(INTL)].groupby('tid').league.agg(lambda s: s.value_counts().index[0])
    p['b_league'] = p.blue_id.map(home).fillna(p.league)
    p['r_league'] = p.red_id.map(home).fillna(p.league)
    rg = pd.concat([pd.read_parquet(f'{D}/role_gold.parquet'), pd.read_parquet(f'{LL}/data/role_gold_new.parquet')]).drop_duplicates('game_id')
    b = p[p.t_start < pd.Timestamp('2024-04-01', tz='UTC')].merge(rg, on='game_id').dropna(subset=['gd15_0'])
    x = b[[f'gd15_{k}' for k in range(5)]].sum(1).to_numpy()[:, None] / 1000
    m = LogisticRegression().fit(x, b.blue_win.astype(int))
    a0, a1 = m.intercept_[0], m.coef_[0, 0]
    f2 = pregame2.build(p, rg, lambda gd: pregame2.sig(a0 + a1 * gd / 1000))
    return p, f2, rg


def reddit_feats(p, rg):
    import build as R
    q = p.copy()
    q['intl'] = q.league.isin(INTL)
    new = q.src.eq('lollive')
    with ThreadPoolExecutor(16) as ex:
        ls = pd.DataFrame(list(ex.map(last_state, q.game_id[new])))
    dm = pd.read_parquet(f'{D}/reddit/dmg.parquet')
    dm = pd.concat([dm, ls.drop(columns='e_gold').dropna()], ignore_index=True)
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
    print(f'new games: e_gold {ls.e_gold.notna().sum()}/{len(ls)}, damage {ls.dmg1.notna().sum() if "dmg1" in ls else 0}/{len(ls)}', flush=True)
    return f


def priors(P):
    import eval as RE
    from eval_pregame2 import BASE, G
    P = P[P.t_start >= BURN].sort_values('t_start').reset_index(drop=True)
    P['age_x_elo'] = P.elo_play_32 * np.exp(-P.patch_age / 7)
    y = P.blue_win.astype(int).to_numpy()
    tr = (P.t_start < SPLIT).to_numpy()

    def fitpred(cols, C, a):
        sc = P.loc[a, cols].std().replace(0, 1)
        m = LogisticRegression(C=C, max_iter=5000).fit(P.loc[a, cols] / sc, y[a])
        return m.decision_function(P[cols] / sc)
    P['prior_new'] = fitpred(RE.V4P, 0.03, tr)
    P['prior_union'] = fitpred(RE.V4P + RE.POST, 0.01, tr)
    P['prior_union_m'] = np.nan
    for m0 in pd.date_range(SPLIT, P.t_start.max() + pd.offsets.MonthBegin(1), freq='MS'):
        cur = ((P.t_start >= m0) & (P.t_start < m0 + pd.offsets.MonthBegin(1))).to_numpy()
        if cur.any():
            P.loc[cur, 'prior_union_m'] = fitpred(RE.V4P + RE.POST, 0.01, (P.t_end < m0).to_numpy())[cur]
    # reproduction check on the study's test games
    s = pd.read_parquet(f'{D}/prior2.parquet').merge(pd.read_parquet(f'{D}/reddit/prior.parquet'), on='game_id')
    c = P.merge(s, on='game_id', suffixes=('', '_study'))
    c = c[c.t_start >= SPLIT]
    print(f'prior check on {len(c)} study test games: |prior_new - study| max {np.abs(c.prior_new - c.prior_new_study).max():.4f} '
          f'median {np.abs(c.prior_new - c.prior_new_study).median():.5f};  union_m vs study prior_both_m max '
          f'{np.abs(c.prior_union_m - c.prior_both_m).max():.4f} median {np.abs(c.prior_union_m - c.prior_both_m).median():.5f}', flush=True)
    return P


def frames(args):
    from ingame_data import one
    from features import raw_frames, window_feats
    gid, src = args
    o = one(gid, src)
    if o is None:
        return None
    d = raw_frames(pd.read_parquet(f'{D}/{src}/{gid}.parquet'))
    if len(d) != len(o):
        return None
    return pd.concat([o.reset_index(drop=True), window_feats(d, o.clock.to_numpy())], axis=1)


def main():
    os.makedirs(f'{D}/v5', exist_ok=True)
    e = ext_table()
    print(f'extended table: {len(e)} games, {e.src.eq("lollive").sum()} from lollive, last {e.t_start.max()}', flush=True)
    p, f2, rg = pregame_all(e)
    fr = reddit_feats(p, rg)
    P = p.merge(f2, on='game_id').merge(fr, on='game_id')
    P = priors(P)
    g = P[P.t_start >= START].copy()
    jobs = [(gid, 'frames60' if s == 'study' else NEWF) for gid, s in zip(g.game_id, g.src)]
    jobs = [j for j in jobs if os.path.exists(f'{D}/{j[1]}/{j[0]}.parquet')]
    with ProcessPoolExecutor(24) as ex:
        F = pd.concat([x for x in ex.map(frames, jobs, chunksize=8) if x is not None], ignore_index=True)
    F = F[F.state != 'finished'].merge(g[['game_id', 'match_id', 'league', 't_start', 'blue_win', 'n_min_games', 'src',
                                          'prior_new', 'prior_union', 'prior_union_m']], on='game_id')
    import eval_v4 as E
    from eval_ingame import STATS, design
    M4 = pickle.load(open(f'{D}/v4/models.pkl', 'rb'))['models']
    M5 = pickle.load(open(f'{D}/v5/models.pkl', 'rb'))
    F['q_v2'] = M4['v2']['model'].predict_proba(design(F.assign(prior_logit=F.prior_new), STATS + ['prior']))[:, 1]
    F['q_v4'] = M4['sel window']['model'].predict_proba(E.X_of(F, None, (M4['sel window']['cols'], 'poly', 'prior_new')))[:, 1]
    F['q_v5'] = M5['model'].predict_proba(E.X_of(F, None, (M5['cols'], 'poly', 'prior_union_m')))[:, 1]
    F['q_v5_fixedprior'] = M5['model'].predict_proba(E.X_of(F, None, (M5['cols'], 'poly', 'prior_union')))[:, 1]
    F.to_parquet(f'{D}/v5/live_period_preds.parquet')
    g.drop(columns=[c for c in g.columns if g[c].dtype == object and c not in ('game_id', 'match_id', 'league', 'blue_id', 'red_id', 'src')]) \
        .to_parquet(f'{D}/v5/live_period_games.parquet')
    y = F.blue_win.astype(int).to_numpy()
    pg = lambda c: pd.Series(ll(y, F[c].to_numpy()), index=F.game_id).groupby(level=0).mean().mean()
    print(f'live period: {F.game_id.nunique()} games, {len(F)} frames; per-game log loss v2 {pg("q_v2"):.4f} v4 {pg("q_v4"):.4f} '
          f'v5 {pg("q_v5"):.4f} (v5 fixed prior {pg("q_v5_fixedprior"):.4f})')


if __name__ == '__main__':
    main()
