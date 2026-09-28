"""Direction 2: calibration and a confidence (uncertainty) output for the in-game model.

A. Calibration of `lr all` on test frames by segment: calibration intercept/slope (logit y ~ a + b logit q; perfect is 0, 1)
   and ECE, by minute bucket, league tier, international, patch age, fewest-games player, PM-linked games.
B. Epistemic uncertainty per frame (the only kind a binary outcome can identify; see PROGRESS, Kim et al. note):
   sd_boot  sd of the logit over B match-bootstrap refits of the prior + lr all pipeline
   sd_dis   sd of the logit across model families (lr all, gbm all, kim mlp, silva lstm, riot/aws + prior)
   Checks: calibration slope by sd tercile; does "integrating out" the uncertainty,
   q' = sigmoid(z / sqrt(1 + pi sd^2 / 8)), lower log loss?
C. Against the market (lag 0 frames): encompassing weight of the model by sd tercile (a useful confidence output
   should put more weight on the model where sd is low).
D. Trading (backtest.py machinery, selection months pick every parameter, eval months scored once):
   B     existing theta gate
   B+lo  theta gate, only frames with sd_dis below the selection median
   Z     z-gate: trade when (q - ask) / sd_dis > k (or (bid - q) / sd_dis > k)
Writes data/ingame_test_preds_conf.parquet (preds plus sd columns) and results/confidence*.csv.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff
from eval_ingame import STATS, PRE, design

D = '../data'
B = int(os.environ.get('BOOT', 20))
TIER1 = {'lck', 'lpl', 'lec', 'lcs', 'lta_north', 'lta_south', 'lcp', 'cblol-brazil', 'pcs', 'vcs', 'ljl-japan'} | INTL


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def calib(y, q):
    """Calibration intercept and slope: logistic regression of y on logit(q)."""
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(q)[:, None], y)
    return m.intercept_[0], m.coef_[0, 0]


def ece(y, q, bins=10):
    b = np.minimum((q * bins).astype(int), bins - 1)
    df = pd.DataFrame(dict(b=b, y=y, q=q)).groupby('b').agg(n=('y', 'size'), y=('y', 'mean'), q=('q', 'mean'))
    return (df.n * (df.y - df.q).abs()).sum() / len(y)


_F = None


def _load():
    f = pd.read_parquet(f'{D}/ingame.parquet')
    return f[(f.t_start >= BURN) & (f.state != 'finished')].reset_index(drop=True)


def boot_one(seed):
    """Refit prior + lr all on a match-bootstrap of the train games; return test logits."""
    global _F
    if _F is None:
        _F = _load()
    f = _F
    tr = (f.t_start < SPLIT).to_numpy()
    r = np.random.default_rng(seed)
    mids = f.match_id[tr].unique()
    pick = pd.Series(r.choice(mids, len(mids))).value_counts()  # match -> multiplicity
    w = f.match_id.map(pick).fillna(0).to_numpy() * tr
    g = f.drop_duplicates('game_id')
    gw = g.match_id.map(pick).fillna(0).to_numpy() * (g.t_start < SPLIT).to_numpy()
    pm = LogisticRegression(max_iter=2000).fit(g[PRE] / 400, g.blue_win.astype(int), sample_weight=gw)
    f = f.assign(prior_logit=pm.decision_function(f[PRE] / 400))
    X = design(f, STATS + ['prior'])
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[w > 0]), f.blue_win.astype(int)[w > 0], sample_weight=w[w > 0])
    return m.decision_function(sc.transform(X[~tr]))


def main():
    f = _load()
    te = (f.t_start >= SPLIT).to_numpy()
    ft = f[te].reset_index(drop=True)
    pr = pd.read_parquet(f'{D}/ingame_test_preds.parquet')
    lc = pd.read_parquet(f'{D}/lit_compare_preds.parquet')
    assert (pr.game_id.to_numpy() == ft.game_id.to_numpy()).all() and (pr.ts.to_numpy() == ft.ts.to_numpy()).all()
    lc = lc.set_index(['game_id', 'ts']).reindex(pd.MultiIndex.from_frame(pr[['game_id', 'ts']]))
    y = pr.blue_win.astype(int).to_numpy()
    z = logit(pr['q_lr all'].to_numpy())

    # ---- B: uncertainty ----
    with ProcessPoolExecutor(min(B, 10)) as ex:
        Z = np.array(list(ex.map(boot_one, range(B))))
    pr['sd_boot'] = Z.std(0)
    fam = np.c_[z, logit(pr['q_gbm all']), logit(lc['q_kim mlp'].to_numpy()), logit(lc['q_silva lstm'].to_numpy()),
                logit(lc['q_riot/aws wp + prior'].to_numpy())]
    pr['sd_dis'] = fam.std(1)
    pr['z_mean_fam'] = fam.mean(1)
    print(f'sd_boot: median {pr.sd_boot.median():.3f} p90 {pr.sd_boot.quantile(.9):.3f};  '
          f'sd_dis: median {pr.sd_dis.median():.3f} p90 {pr.sd_dis.quantile(.9):.3f};  corr {pr.sd_boot.corr(pr.sd_dis):.2f}')

    # segment covariates
    pg = pd.read_parquet(f'{D}/pregame.parquet')[['game_id', 'patch', 'n_min_games']]
    pg['pv'] = pg.patch.str.split('.').str[:2].str.join('.')
    allg = pd.read_parquet(f'{D}/pregame.parquet')[['game_id', 'patch', 't_start']]
    allg['pv'] = allg.patch.str.split('.').str[:2].str.join('.')
    first = allg.groupby('pv').t_start.min()
    pr = pr.merge(pg[['game_id', 'pv', 'n_min_games']], on='game_id', how='left')
    pr['patch_age_d'] = (pr.t_start - pr.pv.map(first)).dt.days
    L0 = pd.read_parquet(f'{D}/market_frames_lag0.parquet')
    pmg = set(L0.game_id)

    # ---- A: calibration by segment ----
    segs = {
        'all': np.ones(len(pr), bool),
        **{f'min {a}-{b}': ((pr.clock >= a * 60) & (pr.clock < b * 60)).to_numpy() for a, b in [(0, 10), (10, 20), (20, 30), (30, 99)]},
        'tier1 league': pr.league.isin(TIER1).to_numpy(), 'tier2 league': ~pr.league.isin(TIER1).to_numpy(),
        'international': pr.league.isin(INTL).to_numpy(),
        'patch age < 7 d': (pr.patch_age_d < 7).to_numpy(), 'patch age >= 7 d': (pr.patch_age_d >= 7).to_numpy(),
        'fewest-games player < 20': (pr.n_min_games < 20).to_numpy(),
        'PM-linked games': pr.game_id.isin(pmg).to_numpy(),
    }
    q = pr['q_lr all'].to_numpy()
    rows = []
    for k, s in segs.items():
        a, b = calib(y[s], q[s])
        rows.append(dict(segment=k, frames=s.sum(), games=pr.game_id[s].nunique(), ll=ll(y[s], q[s]).mean(),
                         ece=ece(y[s], q[s]), cal_int=a, cal_slope=b))
    # sd terciles
    for sd in ('sd_boot', 'sd_dis'):
        t = pd.qcut(pr[sd], 3, labels=['lo', 'mid', 'hi']).to_numpy()
        for lab in ('lo', 'mid', 'hi'):
            s = t == lab
            a, b = calib(y[s], q[s])
            rows.append(dict(segment=f'{sd} {lab}', frames=s.sum(), games=pr.game_id[s].nunique(), ll=ll(y[s], q[s]).mean(),
                             ece=ece(y[s], q[s]), cal_int=a, cal_slope=b))
    A = pd.DataFrame(rows)
    print('\nA/B. calibration of lr all by segment (perfect: int 0, slope 1)')
    print(A.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    A.to_csv(f'{D}/../results/confidence_calib.csv', index=False)

    # integrate out epistemic sd; and the family-mean ensemble
    gm = pr.drop_duplicates('game_id').set_index('game_id').match_id
    base = pd.Series(ll(y, q), index=pr.game_id).groupby(level=0).mean()
    for name, qq in [('shrink by sd_boot', 1 / (1 + np.exp(-z / np.sqrt(1 + np.pi * pr.sd_boot ** 2 / 8)))),
                     ('shrink by sd_dis', 1 / (1 + np.exp(-z / np.sqrt(1 + np.pi * pr.sd_dis ** 2 / 8)))),
                     ('family-mean ensemble', 1 / (1 + np.exp(-pr.z_mean_fam)))]:
        cur = pd.Series(ll(y, np.asarray(qq)), index=pr.game_id).groupby(level=0).mean()
        d, lo, hi = boot_diff(cur.to_numpy(), base.reindex(cur.index).to_numpy(), gm.reindex(cur.index).to_numpy())
        print(f'{name:22s} per-game ll {cur.mean():.4f}  vs lr all {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')

    # ---- C: encompassing weight vs market by sd tercile ----
    m = L0.merge(pr[['game_id', 'ts', 'sd_boot', 'sd_dis']], on=['game_id', 'ts'])
    m = m[(m.mkt > 0.005) & (m.mkt < 0.995)]
    rng = np.random.default_rng(0)
    out = []
    for sd in ('sd_dis', 'sd_boot'):
        m['t'] = pd.qcut(m[sd], 3, labels=['lo', 'mid', 'hi'])
        for lab, x in m.groupby('t', observed=True):
            X = np.c_[logit(x.mkt), logit(x['q_lr all'])]
            yy = x.blue_win.astype(int).to_numpy()
            w = LogisticRegression(C=1e6, max_iter=1000).fit(X, yy).coef_[0]
            mids = x.match_id.unique()
            bs = []
            for _ in range(300):
                sm = rng.choice(mids, len(mids))
                ix = np.concatenate([np.flatnonzero(x.match_id.to_numpy() == k) for k in sm])
                bs.append(LogisticRegression(C=1e6, max_iter=1000).fit(X[ix], yy[ix]).coef_[0, 1])
            out.append(dict(sd=sd, tercile=lab, frames=len(x), w_mkt=w[0], w_model=w[1],
                            lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                            ll_model=ll(yy, x['q_lr all'].to_numpy()).mean(), ll_mkt=ll(yy, x.mkt.to_numpy()).mean()))
    C = pd.DataFrame(out)
    print('\nC. encompassing weights (y ~ logit mkt + logit model) by uncertainty tercile, lag 0')
    print(C.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    C.to_csv(f'{D}/../results/confidence_encompass.csv', index=False)

    pr.to_parquet(f'{D}/ingame_test_preds_conf.parquet')

    # ---- D: trading gates ----
    os.environ['PREDS_SFX'] = '_conf'
    import backtest as bt
    fr, fills = bt.load()
    res = []
    for L in (0, 5):
        qq = bt.quotes(fr, fills, L)
        sel, ev = qq[qq.t_start < bt.SEL_END], qq[qq.t_start >= bt.SEL_END]
        sig = 'q_lr all'

        def gate(d, th, sd_max=np.inf, k=None):
            d = d[d.sd_dis <= sd_max]
            if k is None:
                return bt.run(d, sig, 'B', th)
            d = d.copy()
            s = d.sd_dis.clip(lower=0.05)
            d['_z'] = np.where((d[sig] - d.ask) / s > k, 1.0, np.where((d.bid - d[sig]) / s > k, 0.0, np.nan))
            d = d[d._z.notna()].copy()
            # reuse bt.run's accounting: a signal of 2 (-1) always buys (sells) at the chosen side's fill
            d['_sig'] = np.where(d._z == 1, 2.0, -1.0)
            return bt.run(d, '_sig', 'B', 0.0)

        med = sel.sd_dis.median()
        cand = {
            'B theta': [dict(th=t) for t in bt.THETAS],
            'B theta, sd_dis < sel median': [dict(th=t, sd_max=med) for t in bt.THETAS],
            'Z edge/sd_dis > k': [dict(th=0, k=k) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)],
        }
        for name, grid in cand.items():
            sc = [(bt.summarize(gate(sel, **p), 'sel').get('gross_pp', np.nan), p) for p in grid]
            sc = [s for s in sc if not np.isnan(s[0])]
            best = max(sc, key=lambda s: s[0])
            r = bt.summarize(gate(ev, **best[1]), 'eval')
            res.append(dict(L=L, strat=name, params=str(best[1]), sel_gross_pp=best[0], **r))
        # does profit of the plain theta gate fall with sd? (eval, theta from the first row of this L)
        th = [x for x in res if x['L'] == L and x['strat'] == 'B theta'][0]['params']
        r0 = gate(ev, **eval(th))
        r0 = r0.merge(ev[['game_id', 'clock', 'sd_dis']], on=['game_id', 'clock'], how='left')
        r0['sd_t'] = pd.qcut(r0.sd_dis, 3, labels=['lo', 'mid', 'hi'])
        print(f'\nL={L}: eval trades of the theta gate by sd_dis tercile')
        print(r0.groupby('sd_t', observed=True).gross.agg(n='size', pp=lambda x: 100 * x.mean()).to_string(float_format=lambda x: f'{x:.2f}'))
    R = pd.DataFrame(res).drop(columns='name')
    print('\nD. trading gates (params chosen on selection months, eval Feb-Apr 2026)')
    print(R.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    R.to_csv(f'{D}/../results/confidence_backtest.csv', index=False)


if __name__ == '__main__':
    main()
