"""Draft study (2026-10-16): how much the draft moves each model, how its effect decays in-game, and new draft inputs.

1  pre-game: the draft shift d = z_full - z_pre (draftseries/priors.py) for each prior family (v2 = v2/v4's prior,
   rd = the reddit stack, un = v5/v6's prior): size, value by period / scope, and its calibration slope
   (logit y ~ offset(z_pre) + beta * d; beta 1 = the draft is weighted right).
2  in-game: v4's and v5's frozen in-game logistics on the test frames with the pre-draft prior swapped in; per-minute
   log loss and the prior's effective weight b0 + b1 m + b2 m^2 (m = clock / 10 min).
3  new draft inputs on top of the union prior (monthly refit, C 0.01), each out of sample in time:
     rec    champion-by-role logistic with 30-day half-life sample weights (reddit's draft_wr is unweighted)
     meta   pro pick-rate presence per role (decayed 14-day counts): sum_j log share of blue's pick - red's
     duo    antisymmetric pair one-hot for bot+support and jungle+mid (synergy), monthly logistic
     dmg    damage-profile balance from Data Dragon info.attack / info.magic and tags: |AP share - 0.5|, tanks, ranged
     comfort  player-on-champion games in the last 90 days (log1p), and whether the pick is outside the player's top 5
Usage: python draftseries/draft.py [123]
"""
import os, sys, json, pickle, urllib.request
from collections import defaultdict
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, HERE, os.path.join(SRC, 'reddit'), os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5'), os.path.join(SRC, 'v6')]
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff
import priors as PR

D = os.path.join(SRC, '../data')
FAM = {'v2': 'v2/v4 prior', 'rd': 'reddit stack', 'un': 'v5/v6 union'}
sig = lambda z: 1 / (1 + np.exp(-z))


def cl(a, b, g):
    d = boot_diff(a, b, g)
    return f'{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]'


def fearless():
    return pd.read_parquet(f'{D}/v4/pregame_f.parquet')[['game_id', 'fearless', 'map_no']]


def part1():
    import statsmodels.api as sm
    Q = pd.read_parquet(f'{D}/draftseries/priors.parquet').merge(fearless(), on='game_id', how='left')
    T = Q[(Q.t_start >= SPLIT) & Q.blue_win.notna()].reset_index(drop=True)
    y = T.blue_win.astype(int).to_numpy()
    g = T.match_id.to_numpy()
    print(f'== 1. pre-game draft shift, test {len(T)} games ({T.t_start.min():%Y-%m} .. {T.t_start.max():%Y-%m-%d})')
    for k, nm in FAM.items():
        d = (T[f'{k}_full'] - T[f'{k}_pre']).to_numpy()
        dp = sig(T[f'{k}_full']) - sig(T[f'{k}_pre'])
        a, b = ll(y, sig(T[f'{k}_full'])), ll(y, sig(T[f'{k}_pre']))
        X = sm.add_constant(d)
        r = sm.GLM(y, X, family=sm.families.Binomial(), offset=T[f'{k}_pre'].to_numpy()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(g)[0]})
        hit = ((d > 0) == (y == 1))[np.abs(d) > 0.2].mean()
        print(f'  {nm:14s} SD(d) {d.std():.3f} logit; |dp| median {np.median(np.abs(dp)) * 100:.1f} pp, p90 {np.quantile(np.abs(dp), .9) * 100:.1f} pp, '
              f'>10pp {np.mean(np.abs(dp) > .1):.1%};  LL full - pre {cl(a, b, g)};  slope on d {r.params[1]:.2f} ({r.bse[1]:.2f});  '
              f'draft side wins {hit:.1%} when |d|>0.2')
    a, b = ll(y, sig(T.un_full)), ll(y, sig(T.un_pre))
    print('  union, full - pre by scope:')
    T['half'] = T.t_start.dt.year.astype(str) + 'H' + np.where(T.t_start.dt.month <= 6, '1', '2')
    T['q'] = T.t_start.dt.to_period('Q').astype(str)
    scopes = [('cross-league', T.cross == 1), ('same-league', T.cross == 0), ('international', T.intl == 1),
              ('fearless series', T.fearless == 1), ('non-fearless', T.fearless == 0), ('game 1', T.number == 1), ('game 2+', T.number > 1),
              ('fearless game 3+', (T.fearless == 1) & (T.number >= 3))]
    scopes += [(q, T.q == q) for q in sorted(T.q.unique())]
    for nm, m in scopes:
        m = m.to_numpy()
        if m.sum() > 50:
            sdd = (T.un_full - T.un_pre)[m].std()
            print(f'    {nm:18s} n {m.sum():5d}  SD(d) {sdd:.3f}  {cl(a[m], b[m], g[m])}')


def part2():
    import eval_v4 as E
    print('== 2. in-game: frozen v4 / v5 logistics with the pre-draft prior swapped in (test frames)')
    Q = pd.read_parquet(f'{D}/draftseries/priors.parquet')[['game_id', 'v2_full', 'v2_pre', 'un_full', 'un_pre']]
    f = E.load().merge(pd.read_parquet(f'{D}/reddit/prior.parquet')[['game_id', 'prior_both_m']], on='game_id', how='left')
    f = f[f.t_start >= SPLIT].merge(Q, on='game_id', how='inner').reset_index(drop=True)
    y = f.blue_win.astype(int).to_numpy()
    M4 = pickle.load(open(f'{D}/v4/models.pkl', 'rb'))['models']['sel window']
    M5 = pickle.load(open(f'{D}/v5/models.pkl', 'rb'))
    out = {}
    for nm, M, pc in (('v4', M4, 'v2'), ('v5', M5, 'un')):
        for s in ('full', 'pre'):
            out[f'{nm}_{s}'] = M['model'].predict_proba(E.X_of(f, None, (M['cols'], 'poly', f'{pc}_{s}')))[:, 1]
        # effective prior weight per minute
        lr = M['model'].steps[-1][1]
        sc = M['model'].steps[0][1]
        xc = list(M['xcols'])
        w = {c: lr.coef_[0][xc.index(c)] / sc.scale_[xc.index(c)] for c in ('prior', 'prior_m', 'prior_m2')}
        ws = [w['prior'] + w['prior_m'] * mm + w['prior_m2'] * mm * mm for mm in (0, 1, 2, 3, 4)]
        print(f'  {nm}: prior weight at 0/10/20/30/40 min: ' + ' '.join(f'{x:.2f}' for x in ws))
    f['mb'] = pd.cut(f.clock / 60, [-1, 5, 10, 15, 20, 25, 30, 40, 99], labels=['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-40', '40+'])
    for nm in ('v4', 'v5'):
        a, b = ll(y, out[f'{nm}_full']), ll(y, out[f'{nm}_pre'])
        pa = pd.Series(a).groupby(f.game_id).mean()
        pb = pd.Series(b).groupby(f.game_id).mean()
        g2m = f.drop_duplicates('game_id').set_index('game_id').match_id.reindex(pa.index).to_numpy()
        print(f'  {nm} per-game LL full {pa.mean():.4f} pre {pb.mean():.4f}  full - pre {cl(pa.to_numpy(), pb.reindex(pa.index).to_numpy(), g2m)}')
        dp = np.abs(out[f'{nm}_full'] - out[f'{nm}_pre'])
        row = []
        for b_, m in f.groupby('mb', observed=True).groups.items():
            m = np.asarray(m)
            row.append(f'{b_}: {(a[m] - b[m]).mean():+.4f} |dp| {np.median(dp[m]) * 100:.1f}pp')
        print('    by minute (frame LL full - pre, median |dp|): ' + '; '.join(row))


# ---------- 3. candidate draft inputs ----------

def ddragon():
    fp = f'{D}/draftseries/ddragon_champion.json'
    if not os.path.exists(fp):
        v = json.load(urllib.request.urlopen('https://ddragon.leagueoflegends.com/api/versions.json', timeout=20))[0]
        js = json.load(urllib.request.urlopen(f'https://ddragon.leagueoflegends.com/cdn/{v}/data/en_US/champion.json', timeout=30))
        json.dump(js, open(fp, 'w'))
    js = json.load(open(fp))['data']
    out = {}
    for k, c in js.items():
        i = c['info']
        out[k.lower()] = dict(ap=i['magic'] / max(i['magic'] + i['attack'], 1), tank='Tank' in c['tags'],
                              ranged=c['stats']['attackrange'] > 300, eng=('Tank' in c['tags']) or ('Fighter' in c['tags'] and i['defense'] >= 6))
    return out


def causal_feats(P):
    """meta / dmg / comfort, one causal pass (games applied once they have ended before the next starts)."""
    dd = ddragon()
    miss = set()
    P = P.sort_values('t_start').reset_index(drop=True)
    HLm, HLc = 14.0, 90.0
    cnt = [defaultdict(float) for _ in range(5)]
    tot = [0.0] * 5
    tlast = [None]
    pc = defaultdict(float)
    pcl = {}
    ptop = defaultdict(lambda: defaultdict(float))
    pend = []
    rows = []
    day = lambda ts: ts.value / 86400e9
    import heapq
    for i, r in enumerate(P.itertuples()):
        t = day(r.t_start)
        while pend and pend[0][0] <= t:
            te, j = heapq.heappop(pend)
            q = P.iloc[j]
            # decay meta counts to te
            if tlast[0] is not None:
                f = 0.5 ** ((te - tlast[0]) / HLm)
                for k in range(5):
                    tot[k] *= f
                    for c in cnt[k]:
                        cnt[k][c] *= f
            tlast[0] = te
            for side in ('b', 'r'):
                for k, (c, pl) in enumerate(zip(getattr(q, f'{side}_champs'), getattr(q, f'{side}_players'))):
                    cnt[k][c] += 1
                    tot[k] += 1
                    key = (pl, c)
                    fk = 0.5 ** ((te - pcl[key]) / HLc) if key in pcl else 1.0
                    pc[key] = pc[key] * fk + 1
                    pcl[key] = te
                    ptop[pl][c] += 1
        x = {}
        for side in ('b', 'r'):
            ch, pls = getattr(r, f'{side}_champs'), getattr(r, f'{side}_players')
            K = 170
            x[f'{side}_meta'] = sum(np.log((cnt[k].get(c, 0) + 0.5) / (tot[k] + 0.5 * K)) for k, c in enumerate(ch))
            infos = [dd.get(str(c).lower()) for c in ch]
            for c, inf in zip(ch, infos):
                if inf is None:
                    miss.add(c)
            infos = [z for z in infos if z]
            x[f'{side}_apbal'] = abs(np.mean([z['ap'] for z in infos]) - 0.5) if infos else 0.0
            x[f'{side}_tank'] = sum(z['tank'] for z in infos)
            x[f'{side}_ranged'] = sum(z['ranged'] for z in infos)
            x[f'{side}_eng'] = float(any(z['eng'] for z in infos))
            com, off = 0.0, 0
            for c, pl in zip(ch, pls):
                key = (pl, c)
                v = pc[key] * 0.5 ** ((t - pcl[key]) / HLc) if key in pcl else 0.0
                com += np.log1p(v)
                top = sorted(ptop[pl].items(), key=lambda z: -z[1])[:5]
                off += int(len(top) >= 5 and c not in {z[0] for z in top})
            x[f'{side}_com'], x[f'{side}_off'] = com, off
        rows.append(dict(game_id=r.game_id, meta=x['b_meta'] - x['r_meta'], apbal=x['b_apbal'] - x['r_apbal'],
                         tank=x['b_tank'] - x['r_tank'], ranged=x['b_ranged'] - x['r_ranged'], eng=x['b_eng'] - x['r_eng'],
                         com=x['b_com'] - x['r_com'], off=x['b_off'] - x['r_off']))
        if pd.notna(r.t_end) and pd.notna(r.blue_win):
            heapq.heappush(pend, (day(r.t_end), i))
    if miss:
        print(f'  ddragon: no info for {sorted(miss)[:10]} ({len(miss)})')
    return pd.DataFrame(rows)


def monthly_champ(P, kind):
    """rec: champion-by-role logistic with 30-day half-life weights; duo: antisymmetric pair one-hot (bot+sup, jg+mid)."""
    P = P.sort_values('t_start').reset_index(drop=True)
    cs = sorted(set(c for l in P.b_champs for c in l) | set(c for l in P.r_champs for c in l))
    ix = {c: i for i, c in enumerate(cs)}
    n = len(cs)
    rows, cols, vals = [], [], []
    for i, (b, r) in enumerate(zip(P.b_champs, P.r_champs)):
        if kind == 'rec':
            for j in range(5):
                rows += [i, i]; cols += [j * n + ix[b[j]], j * n + ix[r[j]]]; vals += [1.0, -1.0]
        else:
            for s, (j1, j2) in enumerate(((3, 4), (1, 2))):
                rows += [i, i]; cols += [s * n * n + ix[b[j1]] * n + ix[b[j2]], s * n * n + ix[r[j1]] * n + ix[r[j2]]]; vals += [1.0, -1.0]
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(P), (5 * n) if kind == 'rec' else 2 * n * n))
    y = P.blue_win.fillna(0).astype(int).to_numpy()
    out = np.zeros(len(P))
    day = (P.t_start - P.t_start.min()).dt.total_seconds().to_numpy() / 86400
    for m0 in pd.date_range(P.t_start.min().floor('D') + pd.offsets.MonthBegin(1), P.t_start.max() + pd.offsets.MonthBegin(1), freq='MS'):
        cur = ((P.t_start >= m0) & (P.t_start < m0 + pd.offsets.MonthBegin(1))).to_numpy()
        past = ((P.t_end < m0) & P.blue_win.notna()).to_numpy()
        if not cur.any() or past.sum() < 500:
            continue
        if kind == 'rec':
            t0 = (m0 - P.t_start.min()).total_seconds() / 86400
            w = 0.5 ** ((t0 - day[past]) / 30.0)
            lr = LogisticRegression(C=0.05, max_iter=3000).fit(X[past], y[past], sample_weight=w / w.mean())
        else:
            # columns: only pairs seen >= 3 times in the past (else they are pure noise)
            seen = np.asarray(abs(X[past]).sum(0)).ravel() >= 3
            Xs = X[:, np.where(seen)[0]]
            lr = LogisticRegression(C=0.02, max_iter=3000).fit(Xs[past], y[past])
            out[cur] = lr.decision_function(Xs[cur])
            continue
        out[cur] = lr.decision_function(X[cur])
    return pd.DataFrame({'game_id': P.game_id, f'd_{kind}': out})


def part3():
    import eval as RE
    print('== 3. new draft inputs on top of the union prior (monthly refit, test from 2025-07)')
    P = PR.load()
    fp = f'{D}/draftseries/draft_cands.parquet'
    if os.path.exists(fp):
        Cn = pd.read_parquet(fp)
    else:
        Cn = causal_feats(P).merge(monthly_champ(P, 'rec'), on='game_id').merge(monthly_champ(P, 'duo'), on='game_id')
        Cn.to_parquet(fp)
    P = P.merge(Cn, on='game_id')
    UN = RE.V4P + RE.POST
    arms = {'union': UN, '+rec': UN + ['d_rec'], '+meta': UN + ['meta'], '+duo': UN + ['d_duo'],
            '+dmg': UN + ['apbal', 'tank', 'ranged', 'eng'], '+comfort': UN + ['com', 'off'],
            '+all new': UN + ['d_rec', 'meta', 'd_duo', 'apbal', 'tank', 'ranged', 'eng', 'com', 'off'],
            'union pre-draft': [c for c in UN if c not in PR.DRAFT]}
    T = (P.t_start >= SPLIT) & P.blue_win.notna()
    y = P.blue_win[T].astype(int).to_numpy()
    g = P.match_id[T].to_numpy()
    Z = {k: PR.monthly(P, v, 0.01)[T.to_numpy()] for k, v in arms.items()}
    base = ll(y, sig(Z['union']))
    cr = P.cross[T].to_numpy() == 1
    for k, z in Z.items():
        a = ll(y, sig(z))
        print(f'  {k:16s} LL {a.mean():.4f}  minus union {cl(a, base, g)}  cross {cl(a[cr], base[cr], g[cr])}')
    # standalone signal checks
    for c in ('d_rec', 'meta', 'd_duo', 'apbal', 'tank', 'com', 'off'):
        print(f'    corr({c}, blue_win) test {np.corrcoef(P.loc[T, c], y)[0, 1]:+.3f}; corr with draft_wr {np.corrcoef(P.loc[T, c], P.loc[T, "draft_wr"])[0, 1]:+.2f}')


if __name__ == '__main__':
    parts = sys.argv[1] if len(sys.argv) > 1 else '123'
    for p in parts:
        {'1': part1, '2': part2, '3': part3}[p]()
