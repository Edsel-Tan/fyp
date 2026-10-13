"""The r/leagueoflegends Worlds 2026 model (reddit/build.py) vs our current v4, on our split.

1 pre-game (test 2025-07+): the post's stack in three stages (ratings only / pre-draft / post-draft), each a standardised
  logistic with C chosen on the validation slice (last 91 days of train), as eval_pregame2. Two protocols:
    fixed    fit once on train (ours)
    monthly  refit at each test month start on every game since BURN (the post's "refit monthly")
  plus a LightGBM stack (the post's title says LightGBM), and v4's prior (= v2's prior_new set; v4 left the prior unchanged)
  under both protocols, and the union. Differences vs v4's prior with a match-clustered CI; internationals; by period.
  On Polymarket-linked test games: vs the market's pre-game price (pm_pre, last-30-min VWAP) as in eval_pregame_f.
  Writes data/reddit/prior.parquet (prior_rd, prior_rd_m, prior_both: OOF by match on train as prior2).
2 in-game: v4 (`sel window`, the live tracker's model: STATS + H, S, G, R, prior_new) with the reddit prior in place of
  prior_new, and with both priors; per-game test log loss vs v4. Writes data/ingame_test_preds_rd.parquet.
3 vs the market at lag 0 and 22 s (eval_v4 part 2's protocol).
4 trading backtest (eval_v4 part 3's protocol; gate theta chosen on the selection months; eval months from 2026-02).
Usage: python reddit/eval.py [1234]
"""
import os, sys, pickle
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'v4'))
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff
from eval_pregame2 import BASE, G

D = '../data'
V4P = BASE + sorted({c for k, v in G.items() if not k.startswith('h2h') for c in v})  # prior_new's feature set
RAT = ['kal_z'] + [f'{a}_{k}' for a in ('kal', 'lane', 'dmg') for k in range(5)]
PRE = RAT + ['bt']
POST = PRE + ['draft_wr', 'draft_gd']
CS = (0.003, 0.01, 0.03, 0.1, 1)


def lg(q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def load_pre():
    p = pd.read_parquet(f'{D}/pregame.parquet').merge(pd.read_parquet(f'{D}/pregame2.parquet'), on='game_id') \
         .merge(pd.read_parquet(f'{D}/reddit/feat.parquet'), on='game_id') \
         .merge(pd.read_parquet(f'{D}/v4/pregame_f.parquet')[['game_id', 'pm_pre']], on='game_id', how='left')
    p['age_x_elo'] = p.elo_play_32 * np.exp(-p.patch_age / 7)
    return p[p.t_start >= BURN].sort_values('t_start').reset_index(drop=True)


def part1():
    p = load_pre()
    tr, te = (p.t_start < SPLIT).to_numpy(), (p.t_start >= SPLIT).to_numpy()
    va = tr & (p.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    fit = tr & ~va
    y = p.blue_win.astype(int).to_numpy()

    def fitpred(cols, C, a, b):
        sc = p.loc[a, cols].std().replace(0, 1)
        m = LogisticRegression(C=C, max_iter=5000).fit(p.loc[a, cols] / sc, y[a])
        return m.predict_proba(p.loc[b, cols] / sc)[:, 1], m, sc

    sets = {'v4 prior (prior_new set)': V4P, 'reddit ratings only': RAT, 'reddit pre-draft': PRE,
            'reddit post-draft': POST, 'v4 prior + reddit post-draft': V4P + POST}
    Q, val = {}, {}
    months = pd.date_range(SPLIT, p.t_start.max() + pd.offsets.MonthBegin(1), freq='MS')
    for k, cols in sets.items():
        val[k] = min((log_loss(y[va], fitpred(cols, C, fit, va)[0]), C) for C in CS)
        Q[k] = fitpred(cols, val[k][1], tr, te)[0]
        qm = np.full(len(p), np.nan)
        for m0, m1 in zip(months[:-1], months[1:]):
            cur = ((p.t_start >= m0) & (p.t_start < m1)).to_numpy()
            if cur.any():
                past = (p.t_end < m0).to_numpy()  # every game finished before the month starts
                qm[cur] = fitpred(cols, val[k][1], past, cur)[0]
        Q[k + ' [monthly]'] = qm[te]
    import lightgbm as lgb
    gb = lambda: lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.01, num_leaves=7, min_child_samples=200,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m = gb().fit(p.loc[fit, POST], y[fit], eval_set=[(p.loc[va, POST], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    nt = m.best_iteration_
    m = gb().set_params(n_estimators=nt).fit(p.loc[tr, POST], y[tr])
    Q['reddit post-draft, LightGBM stack'] = m.predict_proba(p.loc[te, POST])[:, 1]

    yt, grp = y[te], p.match_id[te].to_numpy()
    pt = p[te].reset_index(drop=True)
    intl = pt.league.isin(INTL).to_numpy()
    late = (pt.t_start >= pd.Timestamp('2025-09-01', tz='UTC')).to_numpy()
    base = ll(yt, Q['v4 prior (prior_new set)'])
    rows = []
    for k, q in Q.items():
        l = ll(yt, q)
        d = boot_diff(l, base, grp)
        di = boot_diff(l[intl], base[intl], grp[intl])
        rows.append(dict(model=k, C=val.get(k.replace(' [monthly]', ''), (np.nan, np.nan))[1], test_ll=l.mean(), auc=roc_auc_score(yt, q),
                         d=d[0], lo=d[1], hi=d[2], ll_to_aug25=l[~late].mean(), ll_sep25_on=l[late].mean(),
                         intl_ll=l[intl].mean(), d_intl=di[0], intl_ci=f'[{di[1]:+.4f}, {di[2]:+.4f}]'))
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 300)
    print(f'== 1. pre-game: train {tr.sum()} (val {va.sum()})  test {te.sum()} (to Aug-25 {(~late).sum()}, Sep-25 on {late.sum()})  intl {intl.sum()}; lgbm trees {nt}')
    print(R.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    _, m, _ = fitpred(POST, val['reddit post-draft'][1], tr, te)
    print('\nreddit post-draft, standardised coefficients:', pd.Series(m.coef_[0], index=POST).round(3).to_dict())

    print('\n== market pre-game price (Polymarket per-game markets, test games with pm_pre) ==')
    ok = pt.pm_pre.notna().to_numpy() & (pt.pm_pre > 0).to_numpy() & (pt.pm_pre < 1).to_numpy()
    yy, g = yt[ok], grp[ok]
    zm = lg(pt.pm_pre.to_numpy()[ok])
    cand = {'market': pt.pm_pre.to_numpy()[ok], **{k: Q[k][ok] for k in ('v4 prior (prior_new set)', 'v4 prior (prior_new set) [monthly]',
            'reddit post-draft', 'reddit post-draft [monthly]', 'reddit pre-draft', 'v4 prior + reddit post-draft')}}
    for k in ('v4 prior (prior_new set)', 'reddit post-draft'):
        cand[f'50/50 logit blend: market + {k}'] = 1 / (1 + np.exp(-(lg(Q[k][ok]) + zm) / 2))
    bm = ll(yy, cand['market'])
    bv = ll(yy, cand['v4 prior (prior_new set)'])
    print(f'games {ok.sum()}, matches {len(set(g))}')
    for k, q in cand.items():
        l = ll(yy, q)
        d1, d2 = boot_diff(l, bm, g), boot_diff(l, bv, g)
        print(f'  {k:55s} ll {l.mean():.4f}  minus market {d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}]  minus v4 prior {d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}]')
        rows.append(dict(model=f'PM games: {k}', test_ll=l.mean(), d=d2[0], lo=d2[1], hi=d2[2], d_mkt=d1[0], lo_mkt=d1[1], hi_mkt=d1[2]))
    pd.DataFrame(rows).to_csv(f'{D}/../results/reddit_pregame.csv', index=False)

    # priors for the in-game model: OOF by match on train (eval_pregame2's folds), test from the train fit / the monthly refits
    out = p[['game_id']].copy()
    mids = p.match_id[tr].unique()
    fold = pd.Series(np.random.default_rng(0).integers(0, 5, len(mids)), index=mids)
    ftr = p.match_id.map(fold).to_numpy()
    for name, k in (('prior_rd', 'reddit post-draft'), ('prior_both', 'v4 prior + reddit post-draft')):
        z = np.full(len(p), np.nan)
        for j in range(5):
            a, b = tr & (ftr != j), tr & (ftr == j)
            z[b] = lg(fitpred(sets[k], val[k][1], a, b)[0])
        zm_ = z.copy()
        z[te] = lg(Q[k]); zm_[te] = lg(Q[k + ' [monthly]'])
        out[name], out[name + '_m'] = z, zm_
    out.to_parquet(f'{D}/reddit/prior.parquet')


ARMS = {'v4 rd': 'prior_rd', 'v4 rd monthly': 'prior_rd_m', 'v4 union prior': 'prior_both'}


def part2():
    import eval_v4 as E
    from eval_ingame import STATS
    print('\n== 2. in-game: v4 (sel window) with the reddit prior ==')
    M = pickle.load(open(f'{D}/v4/models.pkl', 'rb'))['models']['sel window']
    cols = M['cols']
    f = E.load().merge(pd.read_parquet(f'{D}/reddit/prior.parquet'), on='game_id', how='left')
    assert f[list(ARMS.values())].notna().all().all()
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    ft = f[te][['game_id', 'match_id', 'clock']].reset_index(drop=True)
    ft['y'] = y[te]
    specs = {'v4': (cols, 'prior_new'), **{k: (cols, v) for k, v in ARMS.items()},
             'v4 + rd prior term': (cols + ['prior_rd_x'], 'prior_new')}
    f['prior_rd_x'] = f.prior_rd
    for arm, (cs, pr) in specs.items():
        X = E.X_of(f, None, spec=(cs, 'poly', pr))
        m = E.mk().fit(X[tr], y[tr])
        ft[f'q_{arm}'] = m.predict_proba(X[te])[:, 1]
    pv = pd.read_parquet(f'{D}/ingame_test_preds_v4.parquet')
    assert (pv.game_id.to_numpy() == ft.game_id.to_numpy()).all()
    print(f'  v4 reproduces the stored `sel window`: max |diff| {np.abs(pv["q_sel window"].to_numpy() - ft["q_v4"].to_numpy()).max():.2e}')
    rows = []
    E.compare(ft, [f'q_{a}' for a in specs if a != 'v4'], 'q_v4', 'all test games', rows)
    pt = ft.merge(load_pre()[['game_id', 'league']], on='game_id')
    E.compare(pt[pt.league.isin(INTL)], [f'q_{a}' for a in specs if a != 'v4'], 'q_v4', 'international test games', rows)
    bucket = (ft.clock // 600 * 10).clip(upper=30).astype(int).to_numpy()
    lv = ll(ft.y.to_numpy(), ft.q_v4.to_numpy())
    for a in specs:
        if a != 'v4':
            print(f'    per-frame minus v4 by 10-min bucket {a:20s}', pd.Series(ll(ft.y.to_numpy(), ft[f"q_{a}"].to_numpy()) - lv).groupby(bucket).mean().round(4).to_dict())
    pd.DataFrame(rows).to_csv(f'{D}/../results/reddit_ingame.csv', index=False)
    pv.assign(**{c: ft[c].to_numpy() for c in ft.columns if c.startswith('q_')}).to_parquet(f'{D}/ingame_test_preds_rd.parquet')


SIGS = ['q_v4', 'q_v4 rd', 'q_v4 rd monthly', 'q_v4 union prior', 'q_v4 + rd prior term']


def part3():
    import eval_v4 as E
    import overfit_check as O
    from market_compare import market_at
    print('\n== 3. vs the market (Polymarket per-game markets, test period) ==')
    pr = pd.read_parquet(f'{D}/ingame_test_preds_rd.parquet')
    mkt = pd.read_parquet(f'{D}/pm_game_markets.parquet')
    fills = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event', 'D'])
    fills = fills[fills.condition_id.isin(mkt.condition_id)].merge(mkt[['condition_id', 'seq1_blue']], on='condition_id')
    fills['p_blue'] = np.where(fills.seq1_blue == 1, fills.p_event, 1 - fills.p_event)
    fills['Db'] = np.where(fills.seq1_blue == 1, fills.D, -fills.D)
    fills = {c: g.sort_values('ts') for c, g in fills.groupby('condition_id')}
    mkt = mkt.sort_values('vol', ascending=False).drop_duplicates('game_id')
    base = pr.merge(mkt[['game_id', 'condition_id']], on='game_id')
    rows = []
    for lag in (0, 22):
        fr = market_at(base, fills, lag)
        fr = fr[(fr.mkt_age <= 60) & (fr.mkt > 0) & (fr.mkt < 1)]
        fr = O.mid_proxy(fr, fills, lag)
        fr['y'] = fr.blue_win.astype(int)
        E.compare(fr, SIGS, 'mkt', f'lag {lag}s, all PM frames vs last trade', rows)
        E.compare(fr, SIGS[1:], 'q_v4', f'lag {lag}s, all PM frames, arm minus v4', rows)
        E.compare(fr[fr.mid.notna()], SIGS, 'mid', f'lag {lag}s, two-sided frames vs mid proxy', rows)
    pd.DataFrame(rows).to_csv(f'{D}/../results/reddit_market.csv', index=False)


def part4():
    os.environ['PREDS_SFX'] = '_rd'
    import backtest as B
    print('\n== 4. trading backtest (eval months from 2026-02; gate theta chosen on the selection months) ==')
    fr, fills = B.load()
    rows = []
    for L in (0, 5, 15, 30):
        q = B.quotes(fr, fills, L)
        sel, ev = q[q.t_start < B.SEL_END], q[q.t_start >= B.SEL_END]
        for sig in SIGS:
            rows.append(dict(L=L, sig=sig, strat='clip', **B.summarize(B.run(ev, sig, 'A'), 'eval')))
            best = max(((B.summarize(B.run(sel, sig, 'B', th), 'sel').get('gross_pp', -99), th) for th in B.THETAS),
                       key=lambda z: z[0] if not np.isnan(z[0]) else -99)
            rows.append(dict(L=L, sig=sig, strat=f'gate th={best[1]}', **B.summarize(B.run(ev, sig, 'B', best[1]), 'eval')))
    r = pd.DataFrame(rows).drop(columns='name')
    print(r[['L', 'sig', 'strat', 'n', 'games', 'gross_pp', 'ci_lo', 'ci_hi', 'fee_regime_net_pp']].to_string(
        index=False, float_format=lambda x: f'{x:.2f}'))
    r.to_csv(f'{D}/../results/backtest_reddit.csv', index=False)


if __name__ == '__main__':
    parts = sys.argv[1] if len(sys.argv) > 1 else '1234'
    for k, fn in (('1', part1), ('2', part2), ('3', part3), ('4', part4)):
        if k in parts:
            fn()
