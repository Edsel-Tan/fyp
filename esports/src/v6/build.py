"""v6 = v5 plus the fixes in PROGRESS "Why the market beats the models live" ("What the model lacks, ranked"):

  1 cross-league team strength
      prior   z_x = the union prior + cross-league terms (v6/prior.py, prior_x_m), when it beats v5's union prior on
              validation (XPRIOR below records the choice)
      market  z6 = z_x + has_pm * (w0 + w1 cross) * (logit pm_pre - z_x): the market's pre-game price blended into the prior,
              (w0, w1) fit on the pre-game outcomes of the backtest's selection months (SPLIT .. SEL_END), never later.
              Without a pre-game price z6 = z_x.
      in-game the frozen v5 logistic with z6 in place of its prior; a persistence term (g0 + g1 m) * (z6 - z_x) is kept
              only if a match-grouped CV on the selection months' frames prefers it (the market keeps its opening line,
              v5's prior fades)
  2 fresher information: the live market moves on fights and objectives 10-40 s before the feed shows them, and the feed
      arrives 22-110 s late. Without broadcast capture, the freshest information available is the market's own mid.
      v6m = sigma(a_p + b_p logit(mid) + c_p logit(q_v6)) per game phase p (0-10, 10-20, 20-30, 30+ min), fit on the
      selection months at each latency L (v6/backtest_v6.py, data/v6/comb.pkl); live, the tracker picks the L nearest the
      frame's feed age. v6 alone stays a market-free forecast.
  3 2026 drift: an offset-logistic correction on (gold, kills, towers, alive) x (1, m), refit at each month start on the
      frames of the previous 6 months (test-period games only, so it starts at 2025-08). Tested and off (USE_DRIFT): it is
      +0.0009 on the eval months; 36 variants (inputs, intercept, 3/6/12 months, x0.5 shrink) are in results/v6_drift.log.
Writes data/v6/models.pkl, data/ingame_test_preds_v6.parquet (v5's file + q_v6 and its stages), data/v6/live_period_preds.parquet
(v5/live_period's frames + q_v6) and prints test-split / eval-month / live-period log loss.
Usage: python v6/build.py
"""
import os, sys, pickle
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.model_selection import GroupKFold
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v4'), os.path.join(SRC, 'v5')]
from eval_pregame import SPLIT, ll, boot_diff
import eval_v4 as E

D = '../data'
SEL_END = pd.Timestamp('2026-02-01', tz='UTC')  # backtest.SEL_END
DRIFT = ['gold_k', 'kills', 'towers', 'alive']
DRIFT_MONTHS = 6
USE_DRIFT = os.environ.get('V6_DRIFT', '0') == '1'  # off: no variant of the correction chosen on the selection months helps on the
# eval months or the live period (results/v6_drift.log); q_v6_drift keeps the 6-month version as a reported candidate
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sg = lambda z: 1 / (1 + np.exp(-z))


def choose_prior(PR):
    """union + cross-league terms lost to the union prior on validation (0.6045 vs 0.6038) and test (+0.0005), so v5's union prior (results/v6_prior.log)."""
    return 'prior_x_m' if os.environ.get('V6_XPRIOR', '0') == '1' else 'prior_union_m'


def pm_terms(d, zx):
    has = d.pm_pre.notna().to_numpy()
    dz = np.where(has, lg(d.pm_pre.clip(.01, .99).to_numpy()) - d[zx].to_numpy(), 0.0)
    return dz, dz * d.cross.fillna(0).to_numpy()


def fit_blend(PR, zx):
    g = PR[(PR.t_start >= SPLIT) & (PR.t_start < SEL_END) & PR.pm_pre.notna() & PR.blue_win.notna()].copy()
    a, b = pm_terms(g, zx)
    r = sm.GLM(g.blue_win.astype(int).to_numpy(), np.c_[a, b], family=sm.families.Binomial(), offset=g[zx].to_numpy()).fit()
    print(f'pre-game blend on {len(g)} selection games ({int(g.cross.sum())} cross-league): w0 {r.params[0]:+.3f} ({r.bse[0]:.3f}), '
          f'w1 (cross) {r.params[1]:+.3f} ({r.bse[1]:.3f})')
    return r.params


def z6_of(d, zx, w):
    a, b = pm_terms(d, zx)
    return d[zx].to_numpy() + w[0] * a + w[1] * b


def v5_logit(M5, d, prior_col):
    return lg(M5['model'].predict_proba(E.X_of(d, None, (M5['cols'], 'poly', prior_col)))[:, 1])


def persist_X(d, zx):
    m = d.clock.to_numpy() / 600
    dd = d.z6.to_numpy() - d[zx].to_numpy()
    return np.c_[dd, dd * m]


def fit_persist(S, zx):
    """Persistence term on the selection months' frames (games with a pre-game price): CV says whether it helps."""
    S = S[S.pm_pre.notna()].reset_index(drop=True)
    y, off, X = S.blue_win.astype(int).to_numpy(), S.zb.to_numpy(), persist_X(S, zx)
    gk = GroupKFold(5)
    l0, l1 = [], []
    for a, b in gk.split(X, y, S.match_id):
        r = sm.GLM(y[a], X[a], family=sm.families.Binomial(), offset=off[a]).fit()
        l1.append(ll(y[b], sg(off[b] + X[b] @ r.params)))
        l0.append(ll(y[b], sg(off[b])))
    l0, l1 = np.concatenate(l0).mean(), np.concatenate(l1).mean()
    r = sm.GLM(y, X, family=sm.families.Binomial(), offset=off).fit()
    keep = l1 < l0
    print(f'persistence term on {S.game_id.nunique()} selection games: CV frame LL {l1:.4f} vs {l0:.4f} without -> '
          f'{"kept" if keep else "dropped"}; g0 {r.params[0]:+.3f} g1 {r.params[1]:+.3f}')
    return r.params if keep else np.zeros(2)


def drift_X(d):
    m = d.clock.to_numpy() / 600
    z = d.assign(gold_k=d.gold / 1000)[DRIFT].astype(float).to_numpy()
    return np.c_[np.ones(len(d)), z, z * m[:, None]]


def drift_fit(H):
    r = sm.GLM(H.blue_win.astype(int).to_numpy(), drift_X(H), family=sm.families.Binomial(), offset=H.zp.to_numpy()).fit()
    return r.params


def apply_drift(T, H):
    """Monthly refit on the previous DRIFT_MONTHS months of H (frames with zp and outcomes) -> T.zd."""
    T = T.copy()
    T['zd'] = T.zp
    fits = {}
    for m0 in pd.date_range(T.t_start.min().floor('D') - pd.offsets.MonthBegin(1), T.t_start.max(), freq='MS'):
        cur = ((T.t_start >= m0) & (T.t_start < m0 + pd.offsets.MonthBegin(1))).to_numpy()
        win = H[(H.t_start >= m0 - pd.DateOffset(months=DRIFT_MONTHS)) & (H.t_start < m0)]
        if not cur.any() or win.game_id.nunique() < 300:
            continue
        p = drift_fit(win)
        fits[m0] = p
        T.loc[cur, 'zd'] = T.zp.to_numpy()[cur] + drift_X(T[cur]) @ p
    return T, fits


def per_game(d, c):
    return pd.Series(ll(d.blue_win.astype(int).to_numpy(), sg(d[c].to_numpy())), index=d.game_id).groupby(level=0).mean()


def report(d, cols, base, label):
    g2m = d.drop_duplicates('game_id').set_index('game_id').match_id
    b = per_game(d, base)
    print(f'  [{label}] {d.game_id.nunique()} games: {base} {b.mean():.4f}')
    for c in cols:
        a = per_game(d, c)
        r = boot_diff(a.to_numpy(), b.reindex(a.index).to_numpy(), g2m.reindex(a.index).to_numpy())
        print(f'    {c:10s} {a.mean():.4f}  minus {base} {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]')


def main():
    os.makedirs(f'{D}/v6', exist_ok=True)
    M5 = pickle.load(open(f'{D}/v5/models.pkl', 'rb'))
    PR = pd.read_parquet(f'{D}/v6/priors.parquet')
    zx = choose_prior(PR)
    w = fit_blend(PR, zx)
    PR['z6'] = z6_of(PR, zx, w)
    pcols = ['game_id', 'prior_union_m', 'prior_x_m', 'pm_pre', 'cross', 'z6']

    # ---- study test split (2025-07 .. 2026-09-28)
    pv = pd.read_parquet(f'{D}/ingame_test_preds_v5.parquet')
    import build as B5
    f = B5.load()
    te = f[f.t_start >= SPLIT].reset_index(drop=True)
    assert (te.game_id.to_numpy() == pv.game_id.to_numpy()).all()
    te = te.drop(columns=[c for c in pcols[1:] if c in te]).merge(PR[pcols], on='game_id', how='left')
    assert len(te) == len(pv) and te.z6.notna().all()
    te['z5'] = lg(pv.q_v5.to_numpy())
    te['zx'] = v5_logit(M5, te, zx)
    te['zb'] = v5_logit(M5, te, 'z6')
    sel = te[te.t_start < SEL_END]
    gp = fit_persist(sel, zx)
    te['zp'] = te.zb + persist_X(te, zx) @ gp
    te, fits = apply_drift(te, te)
    print(f'drift refits: {len(fits)} months; last ({max(fits):%Y-%m}) coefs ' + ' '.join(f'{v:+.3f}' for v in fits[max(fits)]))
    for c, z in (('q_v6_x', 'zx'), ('q_v6_pm', 'zb'), ('q_v6_pers', 'zp'), ('q_v6_drift', 'zd'), ('q_v6', 'zd' if USE_DRIFT else 'zp')):
        pv[c] = sg(te[z].to_numpy())
    pv['prior_v6'] = te.z6.to_numpy()
    pv['pm_pre'] = te.pm_pre.to_numpy()
    pv['cross'] = te.cross.to_numpy()
    pv.to_parquet(f'{D}/ingame_test_preds_v6.parquet')
    print('== test split, per-game log loss (stages are cumulative: x = cross-league prior, pm = + market pre-game price, '
          'pers = + persistence, drift = + drift; v6 = the chosen stage) ==')
    stg = ['q_v6_x', 'q_v6_pm', 'q_v6_pers', 'q_v6_drift', 'q_v6']
    T = pv.assign(**{c: lg(pv[c].to_numpy()) for c in stg + ['q_v5']})
    report(T, stg, 'q_v5', 'all test')
    report(T[T.t_start >= SEL_END], stg, 'q_v5', 'eval months 2026-02+')
    report(T[(T.t_start >= SEL_END) & T.pm_pre.notna()], stg, 'q_v5', 'eval months, games with a pre-game price')
    report(T[(T.t_start >= SEL_END) & (T.cross == 1)], stg, 'q_v5', 'eval months, cross-league')
    report(T[T.t_start >= pd.Timestamp('2026-05-01', tz='UTC')], stg, 'q_v5', '2026-05+ (no archive price; drift only)')

    # ---- live period (2026-09-17 .. 10-11; v5/live_period's frames; pm_pre from the captured books)
    F = pd.read_parquet(f'{D}/v5/live_period_preds.parquet')
    F = F.merge(PR[pcols].drop(columns='prior_union_m'), on='game_id', how='left')
    print(f'live period: {F.game_id.nunique()} games, prior_x_m present {F.prior_x_m.notna().mean():.1%}, pm_pre for '
          f'{F.drop_duplicates("game_id").pm_pre.notna().sum()} games')
    F['zx'] = v5_logit(M5, F, zx)
    F['zb'] = v5_logit(M5, F, 'z6')
    F['zp'] = F.zb + persist_X(F, zx) @ gp
    H = pd.concat([te[['game_id', 't_start', 'clock', 'gold', 'kills', 'towers', 'alive', 'blue_win', 'zp']],
                   F[~F.game_id.isin(te.game_id)][['game_id', 't_start', 'clock', 'gold', 'kills', 'towers', 'alive', 'blue_win', 'zp']]])
    F, ff = apply_drift(F, H)
    for c, z in (('q_v6_x', 'zx'), ('q_v6_pm', 'zb'), ('q_v6_pers', 'zp'), ('q_v6_drift', 'zd'), ('q_v6', 'zd' if USE_DRIFT else 'zp')):
        F[c] = sg(F[z].to_numpy())
    F['prior_v6'] = F.z6
    F.drop(columns=['zx', 'zb', 'zp', 'zd']).to_parquet(f'{D}/v6/live_period_preds.parquet')
    T = F.assign(**{c: lg(F[c].to_numpy()) for c in stg + ['q_v5']})
    report(T, stg, 'q_v5', 'live period, all games')
    report(T[T.pm_pre.notna()], stg, 'q_v5', 'live period, games with a captured pre-game price')

    # ---- the live model: the drift correction fit on the last 6 months to the end of H
    end = H.t_start.max()
    live_drift = drift_fit(H[H.t_start >= end - pd.DateOffset(months=DRIFT_MONTHS)])
    pickle.dump(dict(v5=M5, xprior=zx, blend=w, persist=gp, drift=live_drift if USE_DRIFT else None, drift_cols=DRIFT, drift_end=str(end),
                     split=str(SPLIT), sel_end=str(SEL_END), built=str(pd.Timestamp.now(tz='UTC'))), open(f'{D}/v6/models.pkl', 'wb'))
    print(f'live drift fit on {H[H.t_start >= end - pd.DateOffset(months=DRIFT_MONTHS)].game_id.nunique()} games to {end:%Y-%m-%d}: '
          + ' '.join(f'{v:+.3f}' for v in live_drift))


if __name__ == '__main__':
    main()
