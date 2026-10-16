"""v8 candidate, part 1a: time-weighted refits of v7's pre-game prior (2026-10-16).

User idea: patches shift what matters (a top-lane buff should make top-lane ratings count more), so the regression
coefficients should drift. v7's union logistic is refit at each month start on *every* finished game since BURN with equal
weight (an expanding window). Here:

  1 does the reproduction hold: monthly equal-weight refit == data/v7/priors.parquet prior_v7_m
  2 weighted least squares in the logistic sense: sample weight w_i = 0.5 ** (age_i / h), age = refit time - game end (days),
    h in {30, 60, 90, 180, 365, 730} days; weights normalised to mean 1, so C keeps its meaning at full sample.
    Refit cadence monthly (v7's) and weekly. Also a patch-boost variant: games on the latest patch seen get weight x b.
    h (and b) chosen on the selection months 2025-07..2026-01; eval months 2026-02..2026-10-13 scored once.
  3 do the coefficients actually move across patches? Per patch (pv = major.minor, >= 150 games), a logistic of the outcome on
    the v7 prior logit (offset) plus one role family's five role terms (role_elo_k / lane_elo_k / kal_k / lane_k), the
    per-patch slope per role; Cochran's Q = sum (b_p - b_bar)^2 / se_p^2 vs chi2(P - 1), the excess spread tau, and the
    lag-1 autocorrelation of b_p across consecutive patches (if a patch's tilt does not carry into the next patch, no causal
    refit can use it before the new patch has games).
Writes data/v8/prior_decay.parquet (game_id + each arm's logit), results/v8_decay_prior.log (stdout).
Usage: python v8/decay_prior.py
"""
import os, sys, json
import numpy as np, pandas as pd
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC]
from eval_pregame import BURN, SPLIT, INTL, ll, boot_diff

D = os.path.join(SRC, '../data')
SEL_END = pd.Timestamp('2026-02-01', tz='UTC')
sig = lambda z: 1 / (1 + np.exp(-z))
HS = (30, 60, 90, 180, 365, 730)


def load():
    P = pd.read_parquet(f'{D}/v6/prior_feats.parquet').sort_values('t_start').reset_index(drop=True)
    P['age_x_elo'] = P.elo_play_32 * np.exp(-P.patch_age / 7)
    P['intl'] = P.league.isin(INTL)
    F = pd.read_parquet(f'{D}/v7/feat.parquet')
    P = P.merge(F[['game_id', 'kt', 'kl', 'kl_sd', 'kp', 'bt7']], on='game_id', how='left')
    P = P[P.t_start >= BURN].reset_index(drop=True)
    gt = pd.read_parquet(f'{D}/game_table.parquet', columns=['game_id', 'patch'])
    if 'patch' not in P:
        P = P.merge(gt, on='game_id', how='left')
    P['pv'] = P.patch.astype(str).str.split('.').str[:2].str.join('.')
    return P


def fit_at(P, cols, C, t0, h=None, boost=None):
    """Weighted logistic on games ended before t0; returns (model, scale)."""
    a = ((P.t_end < t0) & P.blue_win.notna()).to_numpy()
    X = P.loc[a, cols]
    sc = X.std().replace(0, 1)
    age = (t0 - P.t_end[a]).dt.total_seconds().to_numpy() / 86400
    w = np.ones(a.sum()) if h is None else 0.5 ** (age / h)
    if boost:
        last = P.pv[a].iloc[np.argmax(P.t_end[a].to_numpy())]
        w = w * np.where(P.pv[a].to_numpy() == last, boost, 1.0)
    w = w / w.mean()
    m = LogisticRegression(C=C, max_iter=5000).fit(X / sc, P.blue_win[a].astype(int), sample_weight=w)
    return m, sc


def refit(args):
    P, cols, C, h, boost, cad = args
    out = pd.Series(np.nan, index=P.index)
    freq = 'MS' if cad == 'month' else 'W-MON'
    starts = pd.date_range(SPLIT, P.t_start.max() + pd.Timedelta('32D'), freq=freq)
    starts = starts[starts >= SPLIT]
    if starts[0] > SPLIT:
        starts = starts.insert(0, SPLIT)
    for t0, t1 in zip(starts[:-1], starts[1:]):
        cur = ((P.t_start >= t0) & (P.t_start < t1)).to_numpy()
        if cur.any():
            m, sc = fit_at(P, cols, C, t0, h, boost)
            out[cur] = m.decision_function(P.loc[cur, cols] / sc)
    return out.to_numpy()


def heterogeneity(P, z):
    """Part 3: per-patch role slopes on top of the prior logit z (test + train games since BURN)."""
    fams = {'role_elo': [f'role_elo_{k}' for k in range(5)], 'lane_elo': [f'lane_elo_{k}' for k in range(5)],
            'kal (player Kalman)': [f'kal_{k}' for k in range(5)], 'lane (gd15 resid)': [f'lane_{k}' for k in range(5)]}
    roles = ['top', 'jgl', 'mid', 'bot', 'sup']
    ok = np.isfinite(z) & P.blue_win.notna().to_numpy()
    Q = P[ok].assign(z=z[ok])
    vc = Q.pv.value_counts()
    pats = [p for p in sorted(vc.index, key=lambda s: tuple(int(x) for x in s.split('.'))) if vc[p] >= 150]
    print(f'\n== 3. per-patch role slopes on top of the prior logit ({len(pats)} patches with >= 150 games, '
          f'{Q.pv.isin(pats).sum()} games) ==')
    print('  slope per train SD of the role term; Q = Cochran heterogeneity vs chi2(P-1); tau = excess SD of the slope across patches;')
    print('  ac1 = lag-1 autocorrelation of the patch slopes (does a patch\'s tilt carry into the next one?)')
    rows = []
    for fam, cols in fams.items():
        sd = Q[cols].std()
        B, S = np.full((len(pats), 5), np.nan), np.full((len(pats), 5), np.nan)
        for i, p in enumerate(pats):
            g = Q[Q.pv == p]
            X = sm.add_constant(g[cols] / sd)
            try:
                r = sm.GLM(g.blue_win.astype(int).to_numpy(), X, family=sm.families.Binomial(), offset=g.z.to_numpy()).fit()
                B[i], S[i] = r.params[cols].to_numpy(), r.bse[cols].to_numpy()
            except Exception:
                pass
        # pooled (one slope on top of the prior)
        X = sm.add_constant(Q.loc[Q.pv.isin(pats), cols] / sd)
        gq = Q[Q.pv.isin(pats)]
        rp = sm.GLM(gq.blue_win.astype(int).to_numpy(), X, family=sm.families.Binomial(), offset=gq.z.to_numpy()).fit()
        for k in range(5):
            b, s = B[:, k], S[:, k]
            m = np.isfinite(b) & np.isfinite(s) & (s > 0)
            wts = 1 / s[m] ** 2
            bb = (wts * b[m]).sum() / wts.sum()
            q = (wts * (b[m] - bb) ** 2).sum()
            df = m.sum() - 1
            tau2 = max(0.0, (q - df) / (wts.sum() - (wts ** 2).sum() / wts.sum()))
            bm = b[m]
            ac = np.corrcoef(bm[:-1], bm[1:])[0, 1] if m.sum() > 3 else np.nan
            rows.append(dict(family=fam, role=roles[k], pooled=rp.params[cols[k]], pooled_se=rp.bse[cols[k]], patch_sd=np.std(bm),
                             mean_se=np.sqrt(np.mean(s[m] ** 2)), Q=q, df=df, p=1 - stats.chi2.cdf(q, df), tau=np.sqrt(tau2), ac1=ac))
    r = pd.DataFrame(rows)
    print(r.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    # Holm over the 20 tests
    ps = r.p.sort_values()
    holm = [(p * (len(ps) - i)) for i, p in enumerate(ps)]
    print(f'  smallest Q p {ps.iloc[0]:.4f} (Holm-adjusted {min(1, holm[0]):.3f}); tests with p < 0.05: {(r.p < 0.05).sum()} of {len(r)}')
    return r


def main():
    P = load()
    ch = json.load(open(f'{D}/v7/choice.json'))
    cols, C = ch['cols'], ch['C']
    pr = pd.read_parquet(f'{D}/v7/priors.parquet')[['game_id', 'prior_v7_m']]
    P = P.merge(pr, on='game_id', how='left')
    T = ((P.t_start >= SPLIT) & P.blue_win.notna()).to_numpy()
    print(f'{len(P)} games since BURN; test {T.sum()}; v7 columns {len(cols)}, C {C}')

    arms = {'v7 (monthly, equal weight)': (None, None, 'month')}
    for h in HS:
        arms[f'h {h} d, monthly'] = (h, None, 'month')
        arms[f'h {h} d, weekly'] = (h, None, 'week')
    arms['equal weight, weekly'] = (None, None, 'week')
    for b in (2.0, 4.0):
        arms[f'patch boost x{b:g}, h 365 weekly'] = (365, b, 'week')
        arms[f'patch boost x{b:g}, equal weekly'] = (None, b, 'week')
    with ProcessPoolExecutor(min(24, len(arms))) as ex:
        Z = dict(zip(arms, ex.map(refit, [(P, cols, C, h, b, cad) for h, b, cad in arms.values()])))
    d = np.abs(Z['v7 (monthly, equal weight)'][T] - P.prior_v7_m.to_numpy()[T])
    print(f'1 reproduction: monthly equal-weight refit vs prior_v7_m: max |diff| {np.nanmax(d):.2e}')

    y = P.blue_win.to_numpy()
    g = P.match_id.to_numpy()
    sel = T & (P.t_start < SEL_END).to_numpy()
    ev = T & (P.t_start >= SEL_END).to_numpy()
    cr = (P.cross == 1).to_numpy()
    base = Z['v7 (monthly, equal weight)']
    print('\n== 2. time-weighted refits: per-game log loss minus v7 (match-clustered 95 % CI) ==')
    print(f'  selection months 2025-07..2026-01: {sel.sum()} games; eval months 2026-02..: {ev.sum()} games ({(ev & cr).sum()} cross-league)')
    rows = []
    for k, z in Z.items():
        r = dict(arm=k)
        for nm, m in (('sel', sel), ('eval', ev), ('eval_cross', ev & cr), ('test', T)):
            a, b = ll(y[m].astype(int), sig(z[m])), ll(y[m].astype(int), sig(base[m]))
            dd = boot_diff(a, b, g[m])
            r[nm], r[nm + '_d'], r[nm + '_lo'], r[nm + '_hi'] = a.mean(), dd[0], dd[1], dd[2]
        rows.append(r)
    R = pd.DataFrame(rows)
    for _, r in R.iterrows():
        print(f"  {r.arm:34s} sel {r.sel:.5f} ({r.sel_d:+.5f})   eval {r.eval:.5f} {r.eval_d:+.5f} [{r.eval_lo:+.5f}, {r.eval_hi:+.5f}]"
              f"   eval cross {r.eval_cross_d:+.5f}   all test {r.test_d:+.5f}")
    best = R.iloc[R.sel.idxmin()]
    print(f'  selection-month choice: {best.arm} -> eval {best.eval_d:+.5f} [{best.eval_lo:+.5f}, {best.eval_hi:+.5f}]')
    out = P[['game_id']].copy()
    for k, z in Z.items():
        out[k] = z
    os.makedirs(f'{D}/v8', exist_ok=True)
    out.to_parquet(f'{D}/v8/prior_decay.parquet')
    R.to_csv(f'{D}/../results/v8_decay_prior.csv', index=False)
    json.dump(dict(choice=best.arm), open(f'{D}/v8/prior_decay_choice.json', 'w'))

    # part 3 on the out-of-sample prior (test months) and, for more patches, on all games with the monthly prior
    # from a refit at SPLIT backwards is not causal; use the test months only
    heterogeneity(P[T].reset_index(drop=True), base[T])


if __name__ == '__main__':
    main()
