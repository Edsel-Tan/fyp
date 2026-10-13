"""Why does the Polymarket mid out-forecast the models on live games? (2026-10-12; panel from gap/panel.py)

Sections (match-clustered bootstrap CIs; per-frame log loss on the tradable games' frames with a two-sided book):
  1 timing: v2/v4/v5 vs the mid as of offsets k = -5..+3 min from the frame, and vs the mid interpolated (in logit) to the
    frame's own wall time (mid_same); the model looking 0..3 frames ahead vs the market
  2 by game phase: v5 - mid_same by minute; encompassing logit y ~ z_mkt + z_v5 by phase
  3 pre-game: the market's pre-game price vs the priors, live (cross-league vs same-league) and historical (pm_pre_test, 1,520 games)
  4 counterfactual: frozen v5 with the market's pre-game price (or a 50/50 blend) as its prior
  5 implied weights: OLS of logit(mid_same) and logit(q_v5) on the frame's state, by phase
  6 season drift: offset-logit residual of v5 on gold/kills/... in the study test (2025H2, 2026) and the live period;
    a correction fitted on 2026-01..04 applied to the live period
Writes results/gap_analysis.log (stdout).
"""
import os, sys, pickle
import numpy as np, pandas as pd, statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5')]
from eval_pregame import ll, INTL, SPLIT
import eval_v4 as E

D = '../data'
rng = np.random.default_rng(0)
lg = lambda p: np.log(p / (1 - p))
sg = lambda z: 1 / (1 + np.exp(-z))
PH = [0, 10, 20, 30, 60]


def boot(d, g, n=2000):
    df = pd.DataFrame(dict(d=np.asarray(d), g=np.asarray(g))).groupby('g').d.agg(['sum', 'size'])
    s, c = df['sum'].to_numpy(), df['size'].to_numpy()
    k = rng.integers(0, len(df), (n, len(df)))
    bs = s[k].sum(1) / c[k].sum(1)
    return f'{s.sum() / c.sum():+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]'


def L(x, c):
    return ll(x.blue_win.astype(int).to_numpy(), x[c].clip(1e-4, 1 - 1e-4).to_numpy())


def load():
    F = pd.read_parquet(f'{D}/gap/panel.parquet')
    G = pd.read_parquet(f'{D}/gap/games.parquet').set_index('game_id')
    F['cross'] = F.game_id.map(G.b_league != G.r_league)
    M5 = pickle.load(open(f'{D}/v5/models.pkl', 'rb'))
    F['pm_lg'] = lg(F.pm_pre.clip(.01, .99))
    F['bl_lg'] = .5 * F.pm_lg + .5 * F.prior_union_m
    for nm, c in (('q_v5_pm', 'pm_lg'), ('q_v5_bl', 'bl_lg')):
        ok = F[c].notna()
        F.loc[ok, nm] = M5['model'].predict_proba(E.X_of(F[ok], None, (M5['cols'], 'poly', c)))[:, 1]
    w = (F.ts.astype('int64') // 10 ** 9 % 60).to_numpy() / 60  # the frame's position within its minute
    F['mid_same'] = sg((1 - w) * lg(F['mid_-1'].clip(.005, .995)) + w * lg(F['mid_0'].clip(.005, .995)))
    F['phase'] = pd.cut(F.clock / 60, PH).astype(str)
    return F, G


def timing(F):
    print('== 1. timing ==')
    T = F[F.tradable].dropna(subset=[f'mid_{k}' for k in range(-5, 4)])
    print(f'frames {len(T)}, games {T.game_id.nunique()} (every offset -5..+3 present)')
    base = L(T, 'mid_same')
    for c in ['q_v2', 'q_v4', 'q_v5'] + [f'mid_{k}' for k in range(-5, 4)]:
        print(f'  {c:7s} {L(T, c).mean():.4f}   minus mid_same {boot(L(T, c) - base, T.match_id)}')
    T = T.sort_values(['game_id', 'ts']).copy()
    for j in range(4):
        T[f'q_v5_ahead{j}'] = T.groupby('game_id').q_v5.shift(-j)
    U = T.dropna(subset=[f'q_v5_ahead{j}' for j in range(4)])
    print(f'  frames with 3 later frames ({len(U)}): ' + '  '.join(f'{c} {L(U, c).mean():.4f}' for c in
          [f'q_v5_ahead{j}' for j in range(4)] + ['mid_-1', 'mid_same', 'mid_0', 'mid_1', 'mid_2']))


def phases(F):
    print('\n== 2. by phase (tradable, mid_same) ==')
    T = F[F.tradable].dropna(subset=['mid_same']).copy()
    print(f'frames {len(T)}, games {T.game_id.nunique()};  v5 - mid_same {boot(L(T, "q_v5") - L(T, "mid_same"), T.match_id)};  '
          f'v5 - mid_0 (market up to 60 s newer) {boot(L(T, "q_v5") - L(T, "mid_0"), T.match_id)}')
    T['zm'], T['z5'] = lg(T.mid_same.clip(.01, .99)), lg(T.q_v5.clip(.01, .99))
    for b, x in T.groupby('phase', sort=False):
        r = sm.Logit(x.blue_win.astype(int), sm.add_constant(x[['zm', 'z5']])).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(x.match_id)[0]})
        print(f'  {b:9s} n {len(x):5d}  v5 - mid {boot(L(x, "q_v5") - L(x, "mid_same"), x.match_id)}  mid LL {L(x, "mid_same").mean():.3f};  '
              f'encompassing: market {r.params.zm:+.2f} ({r.bse.zm:.2f}), v5 {r.params.z5:+.2f} ({r.bse.z5:.2f})')
    for k, x in T.groupby('cross'):
        print(f'  {"cross-league" if k else "same-league "} games {x.game_id.nunique():3d}: v5 - mid {boot(L(x, "q_v5") - L(x, "mid_same"), x.match_id)}')
    print('  by league: ' + '; '.join(f'{k} ({x.game_id.nunique()}) {boot(L(x, "q_v5") - L(x, "mid_same"), x.match_id)}'
                                      for k, x in T.groupby('league') if x.game_id.nunique() >= 10))


def pregame(F):
    print('\n== 3. pre-game price vs prior ==')
    G = F[F.tradable & F.pm_pre.notna()].drop_duplicates('game_id').copy()
    y = G.blue_win.astype(int)
    G['L_mkt'], G['L_prior'] = ll(y, G.pm_pre.clip(.01, .99)), ll(y, sg(G.prior_union_m))
    G['L_blend'] = ll(y, sg(G.bl_lg))
    print(f'live, {len(G)} games: market {G.L_mkt.mean():.4f}, v5 prior {G.L_prior.mean():.4f}, blend {G.L_blend.mean():.4f}; '
          f'prior - market {boot(G.L_prior - G.L_mkt, G.match_id)}; corr(logits) {np.corrcoef(G.prior_union_m, G.pm_lg)[0, 1]:.2f}')
    for k, x in G.groupby('cross'):
        print(f'  {"cross" if k else "same "}-league {len(x):3d} games: prior - market {boot(x.L_prior - x.L_mkt, x.match_id)}; blend - market {boot(x.L_blend - x.L_mkt, x.match_id)}')
    r = sm.Logit(y, sm.add_constant(G[['pm_lg', 'prior_union_m']])).fit(disp=0, cov_type='cluster', cov_kwds={'groups': pd.factorize(G.match_id)[0]})
    print(f'  encompassing: market {r.params.pm_lg:+.2f} ({r.bse.pm_lg:.2f}), prior {r.params.prior_union_m:+.2f} ({r.bse.prior_union_m:.2f})')
    big = G[(G.prior_union_m - G.pm_lg).abs() > 0.5]
    print(f'  |prior - market| > 0.5 logit: {len(big)} games, the market\'s side won {(np.sign(big.pm_lg - big.prior_union_m) == np.sign(big.blue_win - .5)).mean():.2f}')
    # historical: the study test games with a Polymarket pre-game price
    t = pd.read_parquet(f'{D}/game_table.parquet', columns=['game_id', 'match_id', 'league', 'blue_id', 'red_id', 'blue_win'])
    tl = pd.concat([t[['blue_id', 'league']].rename(columns={'blue_id': 'tid'}), t[['red_id', 'league']].rename(columns={'red_id': 'tid'})])
    home = tl[~tl.league.isin(INTL)].groupby('tid').league.agg(lambda s: s.value_counts().index[0])
    t['cross'] = t.blue_id.map(home).fillna(t.league) != t.red_id.map(home).fillna(t.league)
    h = pd.read_parquet(f'{D}/v4/pm_pre_test.parquet').merge(pd.read_parquet(f'{D}/reddit/prior.parquet'), on='game_id').merge(t, on='game_id')
    h = h[(h.pm_pre > 0) & (h.pm_pre < 1)]
    y = h.blue_win.astype(int)
    h['d'] = ll(y, sg(h.prior_both_m)) - ll(y, h.pm_pre.clip(.01, .99))
    print(f'historical (study test with a pre-game price, {len(h)} games): prior - market {boot(h.d, h.match_id)}')
    for k, x in h.groupby('cross'):
        print(f'  {"cross" if k else "same "}-league {len(x):4d} games: {boot(x.d, x.match_id)}')
    x = h[h.cross & ~h.league.isin(INTL)]
    print(f'  cross-league, non-international (EMEA Masters, LCP, promotion, ...) {len(x)}: {boot(x.d, x.match_id)}; '
          f'EMEA Masters {boot(h.d[h.league == "emea_masters"], h.match_id[h.league == "emea_masters"])}')
    gm = h.groupby('match_id').agg(d=('d', 'sum'), n=('d', 'size'), c=('cross', 'first'))
    k = rng.integers(0, len(gm), (2000, len(gm)))
    def diff(ix):
        s = gm.iloc[ix]
        return s.d[s.c].sum() / s.n[s.c].sum() - s.d[~s.c].sum() / s.n[~s.c].sum()
    bs = np.array([diff(i) for i in k])
    print(f'  cross minus same: {diff(np.arange(len(gm))):+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]')


def counterfactual(F):
    print('\n== 4. v5 with the market\'s pre-game price as its prior (frozen coefficients) ==')
    T = F[F.tradable].dropna(subset=['mid_same', 'q_v5_pm']).copy()
    for c in ('q_v5', 'q_v5_pm', 'q_v5_bl'):
        print(f'  {c:8s} {L(T, c).mean():.4f}  minus mid_same {boot(L(T, c) - L(T, "mid_same"), T.match_id)}')
    for k, x in list(T.groupby('phase', sort=False)) + [(('cross' if c else 'same'), x) for c, x in T.groupby('cross')]:
        print(f'  {k:9s} ' + '  '.join(f'{c} {boot(L(x, c) - L(x, "mid_same"), x.match_id)}' for c in ('q_v5', 'q_v5_pm', 'q_v5_bl')))


def weights(F):
    print('\n== 5. implied weights: OLS of logit(price) on the state (tradable, 0.03 < mid < 0.97) ==')
    T = F[F.tradable].dropna(subset=['mid_same', 'pm_lg']).copy()
    T = T[(T.mid_same > .03) & (T.mid_same < .97)]
    T['gold_k'] = T.gold / 1000
    X = ['gold_k', 'kills', 'drakes', 'level', 'hp_al', 'alive', 'prior_union_m', 'pm_lg']
    for b, x in T.groupby('phase', sort=False):
        Z = sm.add_constant(x[X].astype(float))
        g = pd.factorize(x.match_id)[0]
        rm = sm.OLS(lg(x.mid_same.clip(.01, .99)), Z).fit()
        r5 = sm.OLS(lg(x.q_v5.clip(.01, .99)), Z).fit()
        rd = sm.OLS(lg(x.mid_same.clip(.01, .99)) - lg(x.q_v5.clip(.01, .99)), Z).fit(cov_type='cluster', cov_kwds={'groups': g})
        print(f'  {b:9s} n {len(x)}: ' + ' | '.join(f'{c} mkt {rm.params[c]:+.2f} v5 {r5.params[c]:+.2f} (t {rd.tvalues[c]:+.1f})' for c in X))


def drift(F):
    print('\n== 6. season drift: offset-logit residual of v5 (coef per unit, x m = clock / 600 s) ==')
    import build as B5
    f = B5.load()
    te = f[f.t_start >= SPLIT].reset_index(drop=True)
    pv = pd.read_parquet(f'{D}/ingame_test_preds_v5.parquet')
    assert len(pv) == len(te) and (pv.game_id.to_numpy() == te.game_id.to_numpy()).all()
    q = pv.q_v5.clip(1e-4, 1 - 1e-4).to_numpy()
    te['z'], te['m'], te['gold_k'] = lg(q), te.clock / 600, te.gold / 1000
    F = F.assign(z=lg(F.q_v5.clip(1e-4, 1 - 1e-4)), m=F.clock / 600, gold_k=F.gold / 1000)
    cols = ['gold_k', 'kills', 'towers', 'alive']

    def Z(d):
        z = d[cols].astype(float).copy()
        for c in cols:
            z[c + '_m'] = d[c] * d.m
        return sm.add_constant(z, has_constant='add')
    y26 = te.t_start >= pd.Timestamp('2026-01-01', tz='UTC')
    for nm, d in (('study test 2025-07..12', te[~y26]), ('study test 2026-01..04+', te[y26]), ('live period', F)):
        r = sm.GLM(d.blue_win.astype(int), Z(d), family=sm.families.Binomial(), offset=d.z).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.game_id)[0]})
        print(f'  {nm:24s} {d.game_id.nunique():5d} games: ' + ' | '.join(
            f'{c} {r.params[c]:+.3f} ({r.bse[c]:.3f}) x m {r.params[c + "_m"]:+.3f} ({r.bse[c + "_m"]:.3f})' for c in cols))
    tr = te[y26 & (te.t_start < F.t_start.min())]
    r = sm.GLM(tr.blue_win.astype(int), Z(tr), family=sm.families.Binomial(), offset=tr.z).fit()
    F['q_v5_26'] = sg(F.z + Z(F) @ r.params)
    print(f'  correction fitted on 2026 study-test games before the live period, applied to all live-period frames: '
          f'{boot(L(F, "q_v5_26") - L(F, "q_v5"), F.match_id)}')
    T = F[F.tradable].dropna(subset=['mid_same'])
    print(f'  tradable, vs mid_same: v5 {boot(L(T, "q_v5") - L(T, "mid_same"), T.match_id)}, corrected {boot(L(T, "q_v5_26") - L(T, "mid_same"), T.match_id)}')


def main():
    pd.set_option('display.width', 250)
    F, G = load()
    timing(F); phases(F); pregame(F); counterfactual(F); weights(F); drift(F)


if __name__ == '__main__':
    main()
