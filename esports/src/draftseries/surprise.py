"""Surprise / dominance terms from earlier games as inputs for later games (2026-10-16).

Per finished game (blue view), from its 60 s frames:
  gold-based (model-free, what the live tracker can compute from its saved feeds):
    g_area    mean over the game of tanh(gold diff / 5k)            (how far ahead, on average)
    lead      share of minutes ahead by > 1k minus share behind by > 1k   (persistence of the lead)
    gd15      gold diff at 15 min / 1k (last frame if shorter)
    gmin/gmax deepest deficit / largest lead (k gold)
  state-based (v5's frozen in-game logistic with the prior set to 0, so it reads the map state only):
    s_area    mean state P(blue) - 0.5
    s_minw    the winner's lowest state P during the game (a deep comeback = low)
  win       y - sigma(z), the result's surprise vs the pre-game prior z (union prior, monthly; from 2024-10 on)
  combos    luck-adjusted result: win_adj = (y - p) * (s_minw-based weight) ; dom_res = y - p where the loser led throughout counts half
Each stat is turned into a surprise vs the pre-game prior: s - E[s | z], E fit (linear in z) on 2024-10..2025-06 games.
Team-level aggregates for a later game, A (blue) minus B (red), over each team's earlier finished games:
    ser_*  same series;   day_*  last 12 h (any match, series included);   d2_* / d7_*  decayed, half-life 2 / 7 days (30-day window)
Tests (offset = the union prior z):
  1 in-sample GLM on 2024-10.. with match-clustered t, and OOS: fit 2024-10..2025-06, score 2025-07..2026-10
  2 the best formulations as extra columns of the union logistic (monthly refit) -> test LL vs the union prior
Writes data/draftseries/dom.parquet (per-game stats) and data/draftseries/surprise_feats.parquet (per-game aggregates).
Usage: python draftseries/surprise.py [build|eval]
"""
import os, sys, glob, pickle, heapq
from collections import defaultdict
import numpy as np, pandas as pd
import statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, HERE, os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5'), os.path.join(SRC, 'reddit')]
from eval_pregame import SPLIT, ll, boot_diff

D = os.path.join(SRC, '../data')
sig = lambda z: 1 / (1 + np.exp(-z))
P0 = pd.Timestamp('2024-10-01', tz='UTC')
STATS = ['g_area', 'lead', 'gd15', 'gmin', 'gmax', 's_area', 's_minw']


def gold_stats(clock, gold):
    """Gold-trajectory stats on a 60 s grid (any frame density). clock in seconds from game start."""
    c, g = np.asarray(clock, float), np.asarray(gold, float)
    o = np.argsort(c)
    c, g = c[o], g[o]
    if len(c) < 3 or c[-1] < 300:
        return None
    grid = np.arange(60.0, c[-1] + 1e-9, 60.0)
    x = np.interp(grid, c, g)
    return dict(g_area=float(np.mean(np.tanh(x / 5000))), lead=float(np.mean(x > 1000) - np.mean(x < -1000)),
                gd15=float(np.interp(min(900.0, c[-1]), c, g) / 1000), gmin=float(x.min() / 1000), gmax=float(x.max() / 1000),
                dur=float(c[-1]))


def state_probs():
    """Per frame: v5's in-game P(blue) with a zero prior (map state only), for the study frames and the live-period frames."""
    import eval_v4 as E
    M5 = pickle.load(open(f'{D}/v5/models.pkl', 'rb'))
    f = E.load()
    f['zero'] = 0.0
    f['sp'] = M5['model'].predict_proba(E.X_of(f, None, (M5['cols'], 'poly', 'zero')))[:, 1]
    a = f[['game_id', 'clock', 'gold', 'sp']]
    L = pd.read_parquet(f'{D}/v6/live_period_preds.parquet')
    L = L[~L.game_id.isin(a.game_id)].copy()
    L['zero'] = 0.0
    L['sp'] = M5['model'].predict_proba(E.X_of(L, None, (M5['cols'], 'poly', 'zero')))[:, 1]
    return pd.concat([a, L[['game_id', 'clock', 'gold', 'sp']]], ignore_index=True)


def build_dom():
    S = state_probs()
    rows = []
    for gid, x in S.groupby('game_id', sort=False):
        r = gold_stats(x.clock.to_numpy(), x.gold.to_numpy())
        if r is None:
            continue
        sp = x.sp.to_numpy()
        r.update(game_id=gid, s_area=float(sp.mean() - 0.5), s_min=float(sp.min()), s_max=float(sp.max()))
        rows.append(r)
    dom = pd.DataFrame(rows)
    dom.to_parquet(f'{D}/draftseries/dom.parquet')
    print(f'dom stats for {len(dom)} games')
    return dom


def table():
    Q = pd.read_parquet(f'{D}/draftseries/priors_ext.parquet')  # un_full from 2024-10 (series robustness run)
    Q = Q[Q.blue_win.notna()].merge(pd.read_parquet(f'{D}/draftseries/dom.parquet'), on='game_id', how='left')
    Q = Q.sort_values('t_start').reset_index(drop=True)
    Q['z'] = Q.un_full
    y = Q.blue_win.to_numpy()
    Q['s_minw'] = np.where(y == 1, Q.s_min, 1 - Q.s_max)  # the winner's lowest state P
    # blue-view signed version of s_minw: + when blue won cleanly, - when red won cleanly
    Q['s_minw_b'] = np.where(y == 1, Q.s_minw, -Q.s_minw)
    return Q


def surprises(Q):
    """Per game, blue-view surprise of each stat vs the prior: s - E[s | z] (linear in z, fit on 2024-10..2025-06)."""
    p = sig(Q.z)
    out = pd.DataFrame({'game_id': Q.game_id})
    out['win'] = Q.blue_win - p
    fitm = (Q.t_start >= P0) & (Q.t_start < SPLIT) & Q.z.notna()
    for s in ['g_area', 'lead', 'gd15', 'gmin', 'gmax', 's_area', 's_minw_b']:
        m = fitm & Q[s].notna()
        b = np.polyfit(Q.z[m], Q[s][m], 1)
        out[s] = Q[s] - np.polyval(b, Q.z)
    # luck-adjusted results: a win counts by how cleanly it was won (winner's lowest state P, 0..~0.5+)
    w = np.clip(Q.s_minw / 0.5, 0, 1).fillna(1.0)
    out['win_adj'] = out.win * w
    out['win_lead'] = out.win * (0.5 + 0.5 * np.abs(Q.lead.fillna(0)))  # persistent leads count fully, flip-flops half
    out['win_dom'] = 0.5 * out.win + 0.5 * out.s_area.fillna(0) * 2
    return out


AGG = {'ser': None, 'day': 0.5, 'd2': 2.0, 'd7': 7.0}
SUR = ['win', 'g_area', 'lead', 'gd15', 'gmin', 'gmax', 's_area', 's_minw_b', 'win_adj', 'win_lead', 'win_dom']


def aggregate(Q, U):
    """A (blue) minus B (red) aggregates of the team-level surprises over earlier finished games."""
    Q = Q.merge(U, on='game_id', how='left', suffixes=('', '_u'))
    hist = defaultdict(list)  # team -> list of (t_end, match_id, vec)
    pend = []
    day = lambda t: t.value / 86400e9
    rows = []
    vec = U.set_index('game_id')[SUR].reindex(Q.game_id).fillna(0.0).to_numpy()
    for i, r in enumerate(Q.itertuples()):
        t = day(r.t_start)
        while pend and pend[0][0] <= t:
            te, j = heapq.heappop(pend)
            g = Q.iloc[j]
            v = vec[j]
            if np.isfinite(g.z):
                hist[g.blue_id].append((te, g.match_id, v))
                hist[g.red_id].append((te, g.match_id, -v))
        x = {'game_id': r.game_id}
        for side, tm in (('A', r.blue_id), ('B', r.red_id)):
            acc = {k: np.zeros(len(SUR)) for k in AGG}
            n = {k: 0 for k in AGG}
            for te, mid, v in reversed(hist[tm]):
                age = t - te
                if age > 30:
                    break
                if mid == r.match_id:
                    acc['ser'] += v; n['ser'] += 1
                if age <= 0.5:
                    acc['day'] += v; n['day'] += 1
                acc['d2'] += v * 0.5 ** (age / 2.0)
                acc['d7'] += v * 0.5 ** (age / 7.0)
            for k in AGG:
                x.update({f'{k}_{s}_{side}': acc[k][j] for j, s in enumerate(SUR)})
            x[f'nday_{side}'] = n['day']
        rows.append(x)
        if pd.notna(r.t_end):
            heapq.heappush(pend, (day(r.t_end), i))
    A = pd.DataFrame(rows)
    for k in AGG:
        for s in SUR:
            A[f'{k}_{s}'] = A[f'{k}_{s}_A'] - A[f'{k}_{s}_B']
    A['nday'] = A.nday_A + A.nday_B
    keep = ['game_id', 'nday'] + [f'{k}_{s}' for k in AGG for s in SUR]
    return A[keep]


def glm(X, cols, off='z'):
    return sm.GLM(X.blue_win.astype(int), sm.add_constant(X[cols], has_constant='add'), family=sm.families.Binomial(),
                  offset=X[off]).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(X.match_id)[0]})


def oos(X, cols, off='z'):
    a, b = X[X.t_start < SPLIT], X[X.t_start >= SPLIT]
    f = sm.GLM(a.blue_win.astype(int), sm.add_constant(a[cols], has_constant='add'), family=sm.families.Binomial(), offset=a[off]).fit()
    f0 = sm.GLM(a.blue_win.astype(int), np.ones((len(a), 1)), family=sm.families.Binomial(), offset=a[off]).fit()
    zb = b[off] + f.params['const'] + sum(f.params[c] * b[c] for c in cols)
    zb0 = b[off] + f0.params.iloc[0]
    y = b.blue_win.astype(int).to_numpy()
    return boot_diff(ll(y, sig(zb)), ll(y, sig(zb0)), b.match_id.to_numpy()), f.params


def evaluate():
    Q = table()
    A = pd.read_parquet(f'{D}/draftseries/surprise_feats.parquet')
    X = Q.merge(A, on='game_id')
    X = X[X.z.notna() & (X.t_start >= P0)].copy()
    print(f'games with a prior from 2024-10: {len(X)}; pre-test (fit) {(X.t_start < SPLIT).sum()}, test {(X.t_start >= SPLIT).sum()}')
    print(f'dominance stats present: {X.lead.notna().mean():.1%} of games')
    # how the per-game stats relate to the result (sanity)
    d = X.dropna(subset=['lead'])
    print('  per-game: corr(lead, win) %.2f, corr(g_area, win) %.2f; winners led > 1k for %.0f%% of minutes on average; '
          'comeback wins (winner state P < 0.2 at some point): %.1f%%' % (np.corrcoef(d.lead, d.blue_win)[0, 1], np.corrcoef(d.g_area, d.blue_win)[0, 1],
                                                                          100 * np.where(d.blue_win == 1, (d.lead + 1) / 2, (1 - d.lead) / 2).mean(),
                                                                          100 * (d.s_minw < 0.2).mean()))
    scopes = {'series k>=2': X.number > 1, 'same-day (any earlier game in 12 h)': X.nday > 0, 'all games': X.number > 0}
    for sn, sm_ in scopes.items():
        S = X[sm_]
        cr = S.cross.to_numpy() == 1
        print(f'\n== {sn}: {len(S)} games ({cr.sum()} cross-league); offset = union prior')
        aggs = ['ser'] if sn == 'series k>=2' else (['day'] if sn.startswith('same') else ['d2', 'd7', 'day'])
        for k in aggs:
            for s in SUR:
                c = f'{k}_{s}'
                r = glm(S, [c])
                o, _ = oos(S, [c])
                Sc = S[cr]
                rc = glm(Sc, [c]) if cr.sum() > 200 else None
                print(f'  {c:14s} coef {r.params[c]:+.3f} (t {r.tvalues[c]:+.1f})  OOS LL {o[0]:+.4f} [{o[1]:+.4f}, {o[2]:+.4f}]'
                      + (f'   cross coef {rc.params[c]:+.3f} (t {rc.tvalues[c]:+.1f})' if rc is not None else ''))
        if sn != 'all games':
            k = aggs[0]
            for cols in ([f'{k}_win', f'{k}_lead'], [f'{k}_win', f'{k}_s_area'], [f'{k}_win', f'{k}_s_minw_b'], [f'{k}_win_adj', f'{k}_g_area']):
                r = glm(S, cols)
                o, _ = oos(S, cols)
                print(f'  {" + ".join(cols):28s} ' + ', '.join(f'{c} {r.params[c]:+.3f} (t {r.tvalues[c]:+.1f})' for c in cols) +
                      f'  OOS LL {o[0]:+.4f} [{o[1]:+.4f}, {o[2]:+.4f}]')
    X.to_parquet(f'{D}/draftseries/surprise_eval.parquet')


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if mode == 'build':
        if not os.path.exists(f'{D}/draftseries/dom.parquet'):
            build_dom()
        Q = table()
        U = surprises(Q)
        A = aggregate(Q, U)
        A.to_parquet(f'{D}/draftseries/surprise_feats.parquet')
        print(f'aggregates for {len(A)} games')
    evaluate()


if __name__ == '__main__':
    main()


def union_test(sets=None, start_fit=pd.Timestamp('2024-11-01', tz='UTC')):
    """Candidate surprise columns inside the union logistic, monthly refit on games since start_fit (the surprise
    features need the 2024-10+ prior), scored on the test months. Base = the union on the same fit rows."""
    import priors as PR, eval as RE
    P = PR.load().merge(pd.read_parquet(f'{D}/draftseries/surprise_feats.parquet'), on='game_id', how='left')
    UN = RE.V4P + RE.POST
    for c in [c for c in P.columns if c[:4] in ('ser_', 'day_', 'd2_w', 'd2_g', 'd2_l', 'd2_s', 'd7_w', 'd7_g', 'd7_s')]:
        P[c] = P[c].fillna(0.0)
        P[f'x_{c}'] = P[c] * P.cross
    P = P[P.t_start >= start_fit].reset_index(drop=True)
    sets = sets or {
        'union (same fit rows)': [],
        '+ser_s_area': ['ser_s_area'], '+day_s_area': ['day_s_area'], '+d2_s_area': ['d2_s_area'],
        '+day_g_area': ['day_g_area'], '+day_lead': ['day_lead'], '+day_win': ['day_win'], '+day_win_dom': ['day_win_dom'],
        '+day_s_area x cross': ['day_s_area', 'x_day_s_area'], '+day_g_area x cross': ['day_g_area', 'x_day_g_area'],
        '+d2_s_area x cross': ['d2_s_area', 'x_d2_s_area'], '+d2_g_area x cross': ['d2_g_area', 'x_d2_g_area'],
        '+day_win x cross': ['day_win', 'x_day_win'],
        '+day (win, s_area, lead) x cross': ['day_win', 'day_s_area', 'day_lead', 'x_day_win', 'x_day_s_area', 'x_day_lead'],
        '+day+d7 s_area x cross': ['day_s_area', 'x_day_s_area', 'd7_s_area', 'x_d7_s_area'],
    }
    T = ((P.t_start >= SPLIT) & P.blue_win.notna()).to_numpy()
    y = P.blue_win[T].astype(int).to_numpy()
    g = P.match_id[T].to_numpy()
    cr = P.cross[T].to_numpy() == 1
    k2 = (P.nday[T].fillna(0) > 0).to_numpy()
    Z = {k: PR.monthly(P, UN + v, 0.01)[T] for k, v in sets.items()}
    base = ll(y, sig(Z['union (same fit rows)']))
    print(f'\n== surprise columns in the union logistic (monthly refit on games since {start_fit:%Y-%m}); test {T.sum()} games, '
          f'{cr.sum()} cross-league, {k2.sum()} with a same-day earlier game')
    for k, z in Z.items():
        a = ll(y, sig(z))
        f = lambda m: boot_diff(a[m], base[m], g[m])
        r, rc, rk = f(np.ones(len(a), bool)), f(cr), f(k2)
        print(f'  {k:34s} {a.mean():.4f}  all {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]  cross {rc[0]:+.4f} [{rc[1]:+.4f}, {rc[2]:+.4f}]  '
              f'same-day {rk[0]:+.4f} [{rk[1]:+.4f}, {rk[2]:+.4f}]')
    return P, Z
