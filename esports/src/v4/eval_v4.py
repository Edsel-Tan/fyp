"""v4 step 4: do the "Missing inputs and feature construction" candidates (features.py, pregame_f.py) add to v2?

Protocol from PROGRESS "Priority": each candidate on the same split vs v2, then vs the market at 0 and 22 s, then live on the same ticks.
v2 = eval_ingame2's `prior_new oof` (STATS + prior, each x (1, m, m^2), StandardScaler + logistic C=1), 0.4255 per-game on test.
Arms (each adds one group to v2 with the same x (1, m, m^2) basis; letters as in features.GROUPS):
  +T transient (inhibs down, baron/elder buff)    +S soul race, drake types       +R respawn seconds left
  +H fight in progress (HP, recent kills)         +D 5-min deltas                 +B bounties
  +I item power spikes (details60)                +L level thresholds             +G relative gold
  +P prior x decidedness (prior_logit x |gold share|; construction 8)
  spline         v2 with a clipped cubic B-spline time basis (knots 1/10/15/20/25/30/40 min, constant beyond) instead of (1, m, m^2)
  prior_f        v2 with the group-F prior (eval_pregame_f.py) in place of prior_new
  all            every group;  all spline: every group on the spline basis;  all window: every group except I (what live can compute)
Parts:
  1 test split: per-game log loss, clustered CI vs v2, by 10-min bucket, and partial effects (logit per SD at 10/20/30 min)
  2 vs the market at lag 0 and 22 s: last trade on all PM frames; the mid proxy on two-sided frames (overfit_check)
  3 trading backtest (backtest.py; gate theta chosen on the selection months) for v2 and the best arms
  4 the live tracker's resolved games rescored from their saved feeds (window-only arms; lollive's clock anchor)
Usage: python v4/eval_v4.py [1234]  (parts; default all). Models: data/v4/models.pkl; preds data/ingame_test_preds_v4.parquet.
"""
import os, sys, pickle, sqlite3, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pregame import BURN, SPLIT, ll, boot_diff
from eval_ingame import STATS, design
from features import GROUPS, WINDOW_ONLY

D = '../data'
LL = os.path.expanduser('~/fyp-pmlive/lollive')
KNOTS = np.array([1, 10, 15, 20, 25, 30, 40], float)[:, None]
rng = np.random.default_rng(0)
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
ALL = [c for g in GROUPS.values() for c in g]
ALLW = [c for k in WINDOW_ONLY for c in GROUPS[k]]
ARMS = {f'+{k}': (STATS + v, 'poly', 'prior_new') for k, v in GROUPS.items()}
ARMS = {'v2': (STATS, 'poly', 'prior_new'), **ARMS,
        '+P': (STATS + ['prdec'], 'poly', 'prior_new'),
        'spline': (STATS, 'spline', 'prior_new'),
        'prior_f': (STATS, 'poly', 'prior_f'),
        'all': (STATS + ALL + ['prdec'], 'poly', 'prior_new'),
        'all spline': (STATS + ALL + ['prdec'], 'spline', 'prior_new'),
        'all window': (STATS + ALLW + ['prdec'], 'poly', 'prior_new')}
_SPL = SplineTransformer(knots=KNOTS, degree=3, extrapolation='constant').fit(KNOTS)


def design_spline(f, cols):
    Bm = _SPL.transform((f.clock.to_numpy() / 60).clip(0, 60)[:, None])
    X = {}
    for c in cols:
        x = (f[c] / (1000 if 'gold' in c else 1) if c in f else f.prior_logit).to_numpy()
        for k in range(Bm.shape[1]):
            X[f'{c}_b{k}'] = x * Bm[:, k]
    return pd.DataFrame(X)


def X_of(f, arm, spec=None):
    cols, basis, prior = spec or ARMS[arm]
    g = f.assign(prior_logit=f[prior])
    g['prdec'] = g.prior_logit * g.gsh.abs() * 10
    return (design if basis == 'poly' else design_spline)(g, cols + ['prior'])


def load():
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')].reset_index(drop=True)
    f = f.merge(pd.read_parquet(f'{D}/prior2.parquet')[['game_id', 'prior_new']], on='game_id', how='left')
    f = f.merge(pd.read_parquet(f'{D}/v4/prior4.parquet'), on='game_id', how='left')
    n = len(f)
    f = f.merge(pd.read_parquet(f'{D}/v4/feat.parquet'), on=['game_id', 'ts'], how='left')
    assert len(f) == n and f[ALL].notna().all().all() and f.prior_f.notna().all()
    f['prdec'] = f.prior_new * f.gsh.abs() * 10
    return f


def pgll(df, col, y='y'):
    return pd.Series(ll(df[y].to_numpy(), df[col].clip(1e-4, 1 - 1e-4).to_numpy()), index=df.game_id).groupby(level=0).mean()


def compare(df, cols, base, label, out=None):
    """Per-game log loss of each col minus base, on the rows where both are present (the live tracker's stored q_v2 is NaN
    on some ticks; scoring different tick sets would bias the difference)."""
    g2m = df.drop_duplicates('game_id').set_index('game_id').match_id
    print(f'  [{label}] games {df.game_id.nunique()} matches {df.match_id.nunique()}')
    for c in cols:
        both = df[df[c].notna() & df[base].notna()]
        a, b = pgll(both, c), pgll(both, base)
        d, lo, hi = boot_diff(a.to_numpy(), b.reindex(a.index).to_numpy(), g2m.reindex(a.index).to_numpy())
        print(f'    {c:16s} {a.mean():.4f}   minus {base} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        if out is not None:
            out.append(dict(scope=label, model=c, ll=a.mean(), base=base, d=d, lo=lo, hi=hi, games=len(a)))


def part1():
    print('== 1. test split (train before 2025-07), per-game log loss ==')
    f = load()
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    ft = f[te][['game_id', 'match_id', 'league', 'ts', 't_start', 'clock', 'blue_win']].reset_index(drop=True)
    ft['y'] = y[te]
    models, eff = {}, {}
    for arm in ARMS:
        t0 = time.time()
        X = X_of(f, arm)
        m = mk().fit(X[tr], y[tr])
        ft[f'q_{arm}'] = m.predict_proba(X[te])[:, 1]
        models[arm] = dict(model=m, cols=ARMS[arm][0], basis=ARMS[arm][1], prior=ARMS[arm][2], xcols=list(X.columns))
        if ARMS[arm][1] == 'poly' and arm.startswith('+'):
            c = pd.Series(m[-1].coef_[0] / m[0].scale_, index=X.columns)
            for k in ARMS[arm][0]:
                if k not in STATS:
                    s = f.loc[tr, k].std() / (1000 if 'gold' in k else 1)
                    eff[k] = [round(float((c[k] + c[k + '_m'] * t + c[k + '_m2'] * t * t) * s), 3) for t in (1, 2, 3)]
        print(f'  fitted {arm:12s} cols {X.shape[1]:4d}  {time.time() - t0:5.0f} s', flush=True)
        del X
    rows = []
    compare(ft, [f'q_{a}' for a in ARMS if a != 'v2'], 'q_v2', 'all test games', rows)
    bucket = (ft.clock // 600 * 10).clip(upper=30).astype(int).to_numpy()
    lv = ll(ft.y.to_numpy(), ft['q_v2'].to_numpy())
    print('\n  per-frame log loss minus v2, by 10-min bucket (frames):', pd.Series(bucket).value_counts().sort_index().to_dict())
    for a in ARMS:
        if a != 'v2':
            print(f'    {a:12s}', pd.Series(ll(ft.y.to_numpy(), ft[f"q_{a}"].to_numpy()) - lv).groupby(bucket).mean().round(4).to_dict())
    print('\n  partial effects of the added terms in the single-group arms (logit per train SD at 10 / 20 / 30 min):')
    for k, v in eff.items():
        print(f'    {k:12s} {v}')
    pd.DataFrame(rows).to_csv(f'{D}/../results/v4_eval.csv', index=False)
    pickle.dump(dict(models=models, split=str(SPLIT), built='2026-10-05'), open(f'{D}/v4/models.pkl', 'wb'))
    pr = pd.read_parquet(f'{D}/ingame_test_preds.parquet')
    assert (pr.game_id.to_numpy() == ft.game_id.to_numpy()).all()
    dv = np.abs(pr['q_lr v2'].to_numpy() - ft.q_v2.to_numpy()).max()
    print(f'  v2 reproduces eval_ingame lr v2: max |diff| {dv:.2e}')
    assert dv < 1e-3
    pr.assign(**{c: ft[c].to_numpy() for c in ft.columns if c.startswith('q_') and c != 'q_v2'}) \
      .to_parquet(f'{D}/ingame_test_preds_v4.parquet')
    return ft


def part2():
    import overfit_check as O
    from market_compare import market_at
    print('\n== 2. vs the market (Polymarket per-game markets, test period) ==')
    pr = pd.read_parquet(f'{D}/ingame_test_preds_v4.parquet')
    mkt = pd.read_parquet(f'{D}/pm_game_markets.parquet')
    fills = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event', 'D'])
    fills = fills[fills.condition_id.isin(mkt.condition_id)].merge(mkt[['condition_id', 'seq1_blue']], on='condition_id')
    fills['p_blue'] = np.where(fills.seq1_blue == 1, fills.p_event, 1 - fills.p_event)
    fills['Db'] = np.where(fills.seq1_blue == 1, fills.D, -fills.D)
    fills = {c: g.sort_values('ts') for c, g in fills.groupby('condition_id')}
    mkt = mkt.sort_values('vol', ascending=False).drop_duplicates('game_id')
    base = pr.merge(mkt[['game_id', 'condition_id']], on='game_id')
    cols = ['q_v2'] + [f'q_{a}' for a in ARMS if a != 'v2']
    rows = []
    for lag in (0, 22):
        fr = market_at(base, fills, lag)
        fr = fr[(fr.mkt_age <= 60) & (fr.mkt > 0) & (fr.mkt < 1)]
        fr = O.mid_proxy(fr, fills, lag)
        fr['y'] = fr.blue_win.astype(int)
        fr['q_v2'] = fr['q_lr v2']
        compare(fr, cols, 'mkt', f'lag {lag}s, all PM frames vs last trade', rows)
        compare(fr, cols[1:], 'q_v2', f'lag {lag}s, all PM frames, arm minus v2', rows)
        two = fr[fr.mid.notna()]
        compare(two, cols, 'mid', f'lag {lag}s, two-sided frames vs mid proxy', rows)
    pd.DataFrame(rows).to_csv(f'{D}/../results/v4_market.csv', index=False)


def part3(arms):
    os.environ['PREDS_SFX'] = '_v4'
    import backtest as B
    print(f'\n== 3. trading backtest (eval months from 2026-02; gate theta chosen on selection months): {arms} ==')
    fr, fills = B.load()
    rows = []
    for L in (0, 5, 15, 30):
        q = B.quotes(fr, fills, L)
        sel, ev = q[q.t_start < B.SEL_END], q[q.t_start >= B.SEL_END]
        for sig in ['q_lr v2'] + [f'q_{a}' for a in arms]:
            rows.append(dict(L=L, sig=sig, strat='clip', **B.summarize(B.run(ev, sig, 'A'), 'eval')))
            best = max(((B.summarize(B.run(sel, sig, 'B', th), 'sel').get('gross_pp', -99), th) for th in B.THETAS),
                       key=lambda z: z[0] if not np.isnan(z[0]) else -99)
            rows.append(dict(L=L, sig=sig, strat=f'gate th={best[1]}', **B.summarize(B.run(ev, sig, 'B', best[1]), 'eval')))
    r = pd.DataFrame(rows).drop(columns='name')
    print(r[['L', 'sig', 'strat', 'n', 'games', 'gross_pp', 'ci_lo', 'ci_hi', 'fee_regime_net_pp']].to_string(
        index=False, float_format=lambda x: f'{x:.2f}'))
    r.to_csv(f'{D}/../results/backtest_v4.csv', index=False)


def part4():
    print('\n== 4. live tracker games, rescored from the saved feeds (window-only arms) ==')
    sys.path.insert(0, f'{LL}/src')
    import live
    from features import raw_frames, window_feats
    M = pickle.load(open(f'{D}/v4/models.pkl', 'rb'))['models']
    MV = pickle.load(open(f'{LL}/data/models.pkl', 'rb'))
    arms = [a for a in ARMS if a not in ('v2', '+I', 'all', 'all spline', 'prior_f') and not (a == 'sel' and 'leg' in ARMS[a][0])]
    c = sqlite3.connect(f'file:{LL}/data/paper.sqlite?mode=ro', uri=True)
    g = pd.read_sql('select * from games where blue_win_rule is not null', c)
    t = pd.read_sql('select * from ticks', c).merge(
        g[['game_id', 'match_id', 'league', 'blue_win_rule', 'team_games_min', 'prior_new']], on='game_id')
    t = t[t.blue_mid.notna() & (t.blue_mid > 0) & (t.blue_mid < 1)].copy()
    out = []
    for gid, tk in t.groupby('game_id'):
        fp = f'{LL}/data/frames60/{gid}.parquet'
        if not os.path.exists(fp):
            continue
        raw = pd.read_parquet(fp)
        o = live.features(raw)
        if o is None:
            continue
        d = raw_frames(raw, min_gold=5000)
        assert len(d) == len(o)
        o = pd.concat([o.reset_index(drop=True), window_feats(d, o.clock.to_numpy())], axis=1).sort_values('ts')
        tk = tk.assign(_t=pd.to_datetime(tk.frame_ts, format='ISO8601')).sort_values('_t')
        j = pd.merge_asof(tk, o.drop(columns='state'), left_on='_t', right_on='ts', direction='backward', suffixes=('', '_f'))
        j = j[j.ts.notna()].reset_index(drop=True)
        j['prior_f'] = np.nan
        j['q_v2re'] = MV['v2']['model'].predict_proba(design(j.assign(prior_logit=j.prior_new), STATS + ['prior']))[:, 1]
        for a in arms:
            j[f'q_{a}'] = M[a]['model'].predict_proba(X_of(j, a))[:, 1]
        out.append(j)
    j = pd.concat(out, ignore_index=True)
    j['y'] = j.blue_win_rule.astype(int)
    j['mid'] = j.blue_mid
    low = j.team_games_min < 20
    ex = j[(j._t == j.ts) & ~low]
    print(f'  rescored ticks {len(j)} of {len(t)}; exact-frame real-prior ticks {len(ex)}; '
          f'recomputed v2 vs stored there: median |diff| {(ex.q_v2re - ex.q_v2).abs().median():.4f}')
    cols = [f'q_{a}' for a in arms]
    rows = []
    compare(j[~low], ['q_v2re'] + cols, 'mid', 'real prior, all ticks (as-of features)', rows)
    compare(j[~low], cols, 'q_v2re', 'real prior, all ticks: arm minus v2 recomputed on the same as-of features', rows)
    compare(ex, ['q_v2'] + cols, 'mid', 'real prior, exact-frame ticks', rows)
    compare(ex, cols, 'q_v2', 'real prior, exact-frame ticks: arm minus stored v2', rows)
    pd.DataFrame(rows).to_csv(f'{D}/../results/v4_live.csv', index=False)
    j.drop(columns=[c for c in j.columns if c.startswith('v3_')], errors='ignore').to_parquet(f'{D}/v4/live_rescored.parquet')


def part5():
    """Groups chosen without looking at test: greedy forward selection on the validation slice (last 91 days of train, fit on
    the rest), adding the group with the best per-game gain while it improves validation log loss by > 0.0001. The chosen set
    ('sel'; 'sel window' without I) is then refit on all of train and scored once on test."""
    print('\n== 5. validation-selected group set ==')
    f = load()
    y = f.blue_win.astype(int).to_numpy()
    tr, te = (f.t_start < SPLIT).to_numpy(), (f.t_start >= SPLIT).to_numpy()
    va = tr & (f.t_start >= SPLIT - pd.Timedelta('91D')).to_numpy()
    fit = tr & ~va
    fv = f[va].reset_index(drop=True)
    cand = list(GROUPS) + ['P']
    cols_of = lambda gs: STATS + [c for g in gs for c in (['prdec'] if g == 'P' else GROUPS[g])]

    def vloss(gs):
        ARMS['_tmp'] = (cols_of(gs), 'poly', 'prior_new')
        X = X_of(f, '_tmp')
        q = mk().fit(X[fit], y[fit]).predict_proba(X[va])[:, 1]
        return pd.Series(ll(y[va], q), index=fv.game_id).groupby(level=0).mean().mean()
    chosen, cur = [], vloss([])
    print(f'  validation games {fv.game_id.nunique()}; v2 {cur:.4f}')
    while True:
        res = {g: vloss(chosen + [g]) for g in cand if g not in chosen}
        if not res:
            break
        g, v = min(res.items(), key=lambda kv: kv[1])
        print(f'  step {len(chosen) + 1}: ' + '  '.join(f'{k} {x - cur:+.4f}' for k, x in sorted(res.items(), key=lambda kv: kv[1])))
        if v > cur - 1e-4:
            break
        chosen.append(g)
        cur = v
    print(f'  chosen: {chosen}')
    ft = f[te][['game_id', 'match_id', 'clock']].reset_index(drop=True)
    ft['y'] = y[te]
    for name, gs in (('v2', []), ('sel', chosen), ('sel window', [g for g in chosen if g != 'I'])):
        ARMS[name] = (cols_of(gs), 'poly', 'prior_new')
        X = X_of(f, name)
        m = mk().fit(X[tr], y[tr])
        ft[f'q_{name}'] = m.predict_proba(X[te])[:, 1]
        if name != 'v2':
            M = pickle.load(open(f'{D}/v4/models.pkl', 'rb'))
            M['models'][name] = dict(model=m, cols=ARMS[name][0], basis='poly', prior='prior_new', xcols=list(X.columns), groups=gs)
            pickle.dump(M, open(f'{D}/v4/models.pkl', 'wb'))
    rows = []
    compare(ft, ['q_sel', 'q_sel window'], 'q_v2', 'all test games', rows)
    pd.DataFrame(rows).to_csv(f'{D}/../results/v4_sel.csv', index=False)
    pr = pd.read_parquet(f'{D}/ingame_test_preds_v4.parquet')
    pr.assign(**{'q_sel': ft['q_sel'].to_numpy(), 'q_sel window': ft['q_sel window'].to_numpy()}) \
      .to_parquet(f'{D}/ingame_test_preds_v4.parquet')
    return chosen


if __name__ == '__main__':
    parts = sys.argv[1] if len(sys.argv) > 1 else '15234'
    if '1' in parts:
        part1()
    if '5' in parts:
        part5()
    ARMS.pop('_tmp', None)
    for a in ('sel', 'sel window'):  # part 5's sets, from the saved models when it did not run in this process
        if a not in ARMS and os.path.exists(f'{D}/v4/models.pkl'):
            m = pickle.load(open(f'{D}/v4/models.pkl', 'rb'))['models'].get(a)
            if m:
                ARMS[a] = (m['cols'], m['basis'], m['prior'])
    if '2' in parts:
        part2()
    if '3' in parts:
        part3(sys.argv[2].split(',') if len(sys.argv) > 2 else ['all', 'all window', 'sel'])
    if '4' in parts:
        part4()
