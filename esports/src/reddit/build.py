"""The r/leagueoflegends "I built a model to predict Worlds 2026" model (2026-10), rebuilt on our data.

The post's recipe, and what each part becomes here (strictly causal, the same single pass as pregame.py: a game's outcome
enters the state only once it has ended before the current game starts):
  Layer 1, player rating engine
    kal_z, kal_k  player win/loss rating with a Kalman filter (an EKF on a Bradley-Terry logit: team strength = mean of the
                  5 players' means), the update weighted by margin of victory (end gold difference) and by internationals;
                  process noise per day, so ratings react to change. kal_z = blue minus red team mean; kal_k = the lane matchup
                  (role k's blue player minus red player).
    lane_k        lane performance residualised on the draft: gold difference at 15:00 vs the lane opponent minus a ridge
                  prediction from all 10 champions (by role, refit monthly on earlier games); player rating = decayed shrunk
                  mean of the residual; blue minus red per role.
    dmg_k         end-of-game damage share residualised the same way (own 5 and enemy 5 champions by role).
  Layer 2, team rating (the "AWS approach with domestic weighting")
    bt            Bradley-Terry: weighted ridge logistic on team and home-league indicators, refit weekly on the last 365 days,
                  decay half-life BT_HL days, internationals weighted BT_INTL, domestic games at full weight.
  Post-draft (the post doesn't say which draft inputs; these are the natural ones its residual models already give)
    draft_wr      logit of a ridge logistic on the 10 champions by role (refit monthly)
    draft_gd      sum over roles of the lane model's draft-expected gold difference at 15:00
The stack (a logistic regression, refit monthly) lives in reddit/eval.py.
Usage: python reddit/build.py [tune]   (tune: grid over the Kalman settings on the validation slice; prints, writes nothing)
Writes data/reddit/feat.parquet.
"""
import os, sys, math, heapq, itertools
from collections import defaultdict
import numpy as np, pandas as pd, scipy.sparse as sp
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import Ridge, LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_pregame import BURN, SPLIT, INTL

D = '../data'
OUT = f'{D}/reddit'
ROLES = ['top', 'jungle', 'mid', 'bottom', 'support']
HL, N0 = 120.0, 5.0  # lane / damage rating: decay half-life (days), shrinkage pseudo-games
BT_HL, BT_INTL, BT_C = 120.0, 2.0, 1.0
KAL = dict(q=1e-3, v0=2.0, w_int=3.0, mov=True)  # best of the `tune` grid (results/reddit_tune.log); v0 and w_int at the grid edge, top 10 within 0.0005


def last_dmg(gid):
    try:
        d = pd.read_parquet(f'{D}/details60/{gid}.parquet', columns=[f'p{i}_dmg' for i in range(1, 11)])
    except Exception:
        return None
    d = d.dropna()
    d = d[d.sum(axis=1) > 1.5]
    if not len(d):
        return None
    return dict(game_id=gid, **{f'dmg{i}': float(d[f'p{i}_dmg'].iloc[-1]) for i in range(1, 11)})


def load():
    p = pd.read_parquet(f'{D}/pregame.parquet')
    tl = pd.concat([p[['blue_id', 'league']].rename(columns={'blue_id': 'tid'}), p[['red_id', 'league']].rename(columns={'red_id': 'tid'})])
    home = tl[~tl.league.isin(INTL)].groupby('tid').league.agg(lambda s: s.value_counts().index[0])
    p['b_league'] = p.blue_id.map(home).fillna(p.league)
    p['r_league'] = p.red_id.map(home).fillna(p.league)
    p['intl'] = p.league.isin(INTL)
    rg = pd.read_parquet(f'{D}/role_gold.parquet')[['game_id'] + [f'gd15_{k}' for k in range(5)]]
    cache = f'{OUT}/dmg.parquet'
    if not os.path.exists(cache):
        with ProcessPoolExecutor(32) as ex:
            dm = pd.DataFrame([x for x in ex.map(last_dmg, p.game_id, chunksize=64) if x is not None])
        dm.to_parquet(cache)
    dm = pd.read_parquet(cache)
    p = p.merge(rg, on='game_id', how='left').merge(dm, on='game_id', how='left')
    return p.sort_values('t_start').reset_index(drop=True)


class Champs:
    """Antisymmetric champion-by-role design: role j's blue champion +1, red champion -1 (draft_wr, lane gd15)
    and per-side (own / enemy) design for a player's damage share."""
    def __init__(self, p):
        cs = sorted(set(c for l in p.b_champs for c in l) | set(c for l in p.r_champs for c in l))
        self.ix = {c: i for i, c in enumerate(cs)}
        self.n = len(cs)

    def anti(self, bc, rc):
        rows, cols, vals = [], [], []
        for i, (b, r) in enumerate(zip(bc, rc)):
            for j in range(5):
                rows += [i, i]; cols += [j * self.n + self.ix[b[j]], j * self.n + self.ix[r[j]]]; vals += [1.0, -1.0]
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(bc), 5 * self.n))

    def side(self, own, enemy):
        rows, cols, vals = [], [], []
        for i, (o, e) in enumerate(zip(own, enemy)):
            for j in range(5):
                rows += [i, i]; cols += [j * self.n + self.ix[o[j]], (5 + j) * self.n + self.ix[e[j]]]; vals += [1.0, 1.0]
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(own), 10 * self.n))


def draft_models(p, ch):
    """Monthly refits (on games ended before each month start) of: draft win model, lane gd15 per role, damage share per role.
    Returns per-game draft_wr, draft-expected gd15 per role and damage-share expectations per player slot."""
    months = pd.date_range(p.t_start.min().floor('D') - pd.offsets.MonthBegin(1), p.t_start.max() + pd.offsets.MonthBegin(1), freq='MS', tz='UTC')
    Xa = ch.anti(p.b_champs, p.r_champs)
    Xb = ch.side(p.b_champs, p.r_champs)  # blue players' view
    Xr = ch.side(p.r_champs, p.b_champs)  # red players' view
    y = p.blue_win.astype(int).to_numpy()
    out = dict(draft_wr=np.zeros(len(p)), **{f'e_gd_{k}': np.zeros(len(p)) for k in range(5)},
               **{f'e_dmg{i}': np.full(len(p), 0.2) for i in range(1, 11)})
    for m0, m1 in zip(months[:-1], months[1:]):
        cur = ((p.t_start >= m0) & (p.t_start < m1)).to_numpy()
        if not cur.any():
            continue
        past = (p.t_end < m0).to_numpy()
        if past.sum() < 500:  # first months: no model yet (they are rating burn-in anyway)
            continue
        lr = LogisticRegression(C=0.05, max_iter=3000).fit(Xa[past], y[past])
        out['draft_wr'][cur] = lr.decision_function(Xa[cur])
        for k in range(5):
            ok = past & p[f'gd15_{k}'].notna().to_numpy()
            r = Ridge(alpha=30.0).fit(Xa[ok], p.loc[ok, f'gd15_{k}'].to_numpy() / 1000)
            out[f'e_gd_{k}'][cur] = r.predict(Xa[cur]) * 1000
            okd = past & p[f'dmg{k + 1}'].notna().to_numpy()
            X = sp.vstack([Xb[okd], Xr[okd]])
            t = np.r_[p.loc[okd, f'dmg{k + 1}'].to_numpy(), p.loc[okd, f'dmg{k + 6}'].to_numpy()]
            r = Ridge(alpha=30.0).fit(X, t)
            out[f'e_dmg{k + 1}'][cur] = r.predict(Xb[cur])
            out[f'e_dmg{k + 6}'][cur] = r.predict(Xr[cur])
    return pd.DataFrame(out)


def bt_ratings(p):
    """Weekly-refit Bradley-Terry (team + home league indicators, weighted ridge logistic). Returns bt (blue minus red logit)."""
    teams = sorted(set(p.blue_id) | set(p.red_id))
    lgs = sorted(set(p.b_league) | set(p.r_league))
    ti, li = {t: i for i, t in enumerate(teams)}, {l: len(teams) + i for i, l in enumerate(lgs)}
    nc = len(teams) + len(lgs)
    rows, cols, vals = [], [], []
    for i, g in enumerate(p.itertuples()):
        rows += [i, i]; cols += [ti[g.blue_id], ti[g.red_id]]; vals += [1.0, -1.0]
        if g.b_league != g.r_league:
            rows += [i, i]; cols += [li[g.b_league], li[g.r_league]]; vals += [1.0, -1.0]
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(p), nc))
    y = p.blue_win.astype(int).to_numpy()
    tday = (p.t_start - p.t_start.min()).dt.total_seconds().to_numpy() / 86400
    eday = (p.t_end - p.t_start.min()).dt.total_seconds().to_numpy() / 86400
    w_int = np.where(p.intl, BT_INTL, 1.0)
    bt = np.zeros(len(p))
    for w0 in np.arange(0, tday.max() + 7, 7.0):
        cur = (tday >= w0) & (tday < w0 + 7)
        if not cur.any():
            continue
        past = (eday < w0) & (eday >= w0 - 365)
        if past.sum() < 200 or len(set(y[past])) < 2:
            continue
        sw = w_int[past] * 0.5 ** ((w0 - eday[past]) / BT_HL)
        m = LogisticRegression(C=BT_C, max_iter=3000).fit(X[past], y[past], sample_weight=sw / sw.mean())
        coef = m.coef_[0]
        Xc = X[cur]
        bt[cur] = Xc @ coef  # side intercept excluded: a rating, not a forecast
    return bt


def player_pass(p, dr, kal):
    """Kalman player ratings + residual lane / damage ratings, one causal pass."""
    q, v0, w_int, mov = kal['q'], kal['v0'], kal['w_int'], kal['mov']
    mu, var, tlast = {}, {}, {}
    lgp = defaultdict(list)
    sl, nl, tl_ = defaultdict(float), defaultdict(float), {}
    sd, nd, td = defaultdict(float), defaultdict(float), {}
    days = ((p.t_start - p.t_start.min()).dt.total_seconds() / 86400).to_numpy()
    edays = ((p.t_end - p.t_start.min()).dt.total_seconds() / 86400).to_numpy()
    pend, rows = [], []
    B, R = p.b_players.to_numpy(), p.r_players.to_numpy()

    def get(pl, lg, now):
        if pl not in mu:
            cand = [mu[x] for x in lgp[lg][-200:]]
            mu[pl], var[pl], tlast[pl] = (float(np.mean(cand)) if cand else 0.0), v0, now
            lgp[lg].append(pl)
        v = min(var[pl] + q * max(0.0, now - tlast[pl]), v0)
        return mu[pl], v

    def dec(s, n, t, k, now):
        if k not in t:
            return 0.0, 0.0
        f = 0.5 ** ((now - t[k]) / HL)
        return s[k] * f, n[k] * f

    def add(s, n, t, k, x, now):
        f = 0.5 ** ((now - t[k]) / HL) if k in t else 1.0
        s[k] = s[k] * f + x; n[k] = n[k] * f + 1; t[k] = now

    def apply(i):
        g = p.iloc[i]
        now = edays[i]
        y = float(g.blue_win)
        bs = [get(pl, g.b_league, now) for pl in B[i]]
        rs = [get(pl, g.r_league, now) for pl in R[i]]
        z = np.mean([m for m, _ in bs]) - np.mean([m for m, _ in rs])
        V = (sum(v for _, v in bs) + sum(v for _, v in rs)) / 25
        pr = 1 / (1 + math.exp(-z))
        w = (w_int if g.intl else 1.0)
        if mov and not np.isnan(g.e_gold):
            w *= min(max(math.log1p(abs(g.e_gold) / 1000) / math.log(6), 0.3), 1.6)
        den = 1 + w * V * pr * (1 - pr)
        for pls, st, sgn in ((B[i], bs, 1), (R[i], rs, -1)):
            for pl, (m, v) in zip(pls, st):
                mu[pl] = m + sgn * w * v * 0.2 * (y - pr) / den
                var[pl] = max(v - w * v * v * 0.04 * pr * (1 - pr) / den, 1e-4)
                tlast[pl] = now
        for k in range(5):
            gd = g[f'gd15_{k}']
            if not np.isnan(gd):
                r = (gd - dr[f'e_gd_{k}'][i]) / 1000
                add(sl, nl, tl_, B[i][k], r, now); add(sl, nl, tl_, R[i][k], -r, now)
            for j, pl in ((k + 1, B[i][k]), (k + 6, R[i][k])):
                x = g[f'dmg{j}']
                if not np.isnan(x):
                    add(sd, nd, td, pl, (x - dr[f'e_dmg{j}'][i]) * 100, now)

    for i in range(len(p)):
        now = days[i]
        while pend and pend[0][0] <= p.t_start.iloc[i]:
            apply(heapq.heappop(pend)[1])
        g = p.iloc[i]
        f = dict(game_id=g.game_id)
        bs = [get(pl, g.b_league, now) for pl in B[i]]
        rs = [get(pl, g.r_league, now) for pl in R[i]]
        f['kal_z'] = np.mean([m for m, _ in bs]) - np.mean([m for m, _ in rs])
        f['kal_sd'] = math.sqrt((sum(v for _, v in bs) + sum(v for _, v in rs)) / 25)
        for k in range(5):
            f[f'kal_{k}'] = bs[k][0] - rs[k][0]
            s1, n1 = dec(sl, nl, tl_, B[i][k], now); s2, n2 = dec(sl, nl, tl_, R[i][k], now)
            f[f'lane_{k}'] = s1 / (n1 + N0) - s2 / (n2 + N0)
            s1, n1 = dec(sd, nd, td, B[i][k], now); s2, n2 = dec(sd, nd, td, R[i][k], now)
            f[f'dmg_{k}'] = s1 / (n1 + N0) - s2 / (n2 + N0)
        rows.append(f)
        heapq.heappush(pend, (p.t_end.iloc[i], i))
    return pd.DataFrame(rows)


def _kal_z(p, dr, k):
    return player_pass(p, dr, k).kal_z.to_numpy()


def tune(p, dr):
    """Validation grid for the Kalman settings: kal_z alone, logistic fit on [BURN, val start), scored on the val slice."""
    va0 = SPLIT - pd.Timedelta('91D')
    fit = ((p.t_start >= BURN) & (p.t_start < va0)).to_numpy()
    va = ((p.t_start >= va0) & (p.t_start < SPLIT)).to_numpy()
    y = p.blue_win.astype(int).to_numpy()
    grid = [dict(q=q, v0=v0, w_int=wi, mov=mov) for q, v0, wi, mov in
            itertools.product((1e-4, 3e-4, 1e-3, 3e-3), (0.5, 1.0, 2.0), (1.0, 2.0, 3.0), (False, True))]
    with ProcessPoolExecutor(24) as ex:
        zs = list(ex.map(_kal_z, [p] * len(grid), [dr] * len(grid), grid))
    res = []
    for k, z in zip(grid, zs):
        x = z[:, None]
        m = LogisticRegression(C=1e6).fit(x[fit], y[fit])
        qv = np.clip(m.predict_proba(x[va])[:, 1], 1e-6, 1 - 1e-6)
        res.append(dict(**k, val_ll=-np.mean(y[va] * np.log(qv) + (1 - y[va]) * np.log(1 - qv))))
    r = pd.DataFrame(res).sort_values('val_ll')
    print(r.head(10).to_string(index=False))
    return r.iloc[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    p = load()
    print(f'games {len(p)}; gd15 {p.gd15_0.notna().mean():.3f}; dmg {p.dmg1.notna().mean():.3f}', flush=True)
    ch = Champs(p)
    dr = draft_models(p, ch)
    print('draft models done', flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'tune':
        best = tune(p, dr)
        print('best', best.to_dict())
        return
    f = player_pass(p, dr, KAL)
    f['bt'] = bt_ratings(p)
    f['draft_wr'] = dr.draft_wr.to_numpy()
    f['draft_gd'] = sum(dr[f'e_gd_{k}'] for k in range(5)).to_numpy() / 1000
    f.to_parquet(f'{OUT}/feat.parquet')
    print(f.describe().T[['mean', 'std']].round(3).to_string())


if __name__ == '__main__':
    main()
