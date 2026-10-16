"""v7 pre-game rating features (2026-10-16). Self-contained (numpy / pandas / sklearn), copied verbatim into the live tracker
(~/fyp-pmlive/lollive/src/v7feat.py), so the study and the tracker compute the same numbers.

Input: a game table sorted by t_start with game_id, match_id, t_start, t_end, blue_id, red_id, b_league, r_league, league,
blue_win (NaN for live rows), intl (bool), g_area (the game's mean tanh(gold diff / 5k), NaN when unknown / live).
A game's own result never enters its own features: games are applied once they have ended before the current game starts.

1 team + league Kalman (PROGRESS "Draft and series updating", change 2: cross-league series should move more)
    Each team has a within-league rating theta (starts at 0 = its league's average) and each home league an offset mu.
    P(blue) = sigma(theta_b - theta_r + mu_Lb - mu_Lr); same-league games cancel mu. A diagonal EKF (Glicko-style) updates
    every term by its own variance, and variances grow with time (q per day, capped at v0). Leagues that rarely meet keep a
    large variance, so the first cross-league games move the league offset (and both teams) a lot, and the series after it
    keeps moving until the variance shrinks. Features: kt (theta diff), kl (mu diff), kl_sd, kp (the filter's own logit).
2 online Bradley-Terry (change 3: BT was frozen within each week)
    The weekly weighted ridge logistic (reddit/build.bt_ratings: team + home-league indicators, 365 days, half-life 120 d,
    internationals x2, C 1) is kept as the anchor; inside the week every finished game updates it by one exact Newton step
    on its own likelihood term (Sherman-Morrison on the fit's Hessian), so bt7 moves after every game.
3 same-day dominance surprise (the user's surprise term)
    For each finished game, s = g_area - E[g_area | kp] (linear in the Kalman logit, fit on games before E_END), team view
    signed. day_dom = A's minus B's sum of s over their earlier finished games in the last 12 h (series games included).
    Also ser_dom (same series only) and d2_dom (2-day half-life decayed sum, 30-day window) for comparison.
"""
import math, heapq
from collections import defaultdict
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

# chosen on the validation slice (results/v7_kgrid.log): fast team ratings (variance refills within ~2 days), slow league offsets,
# updates on the soft result with lam 0.5 (0.25 is within 0.00004; the less extreme value is kept)
KAL7 = dict(q_t=0.5, q_l=1e-4, v_t=1.0, v_l=1.0, w_int=2.0, side=0.12, lam=0.5, kd=3.0)
BT_HL, BT_INTL, BT_C = 120.0, 2.0, 1.0
E_END = pd.Timestamp('2025-07-01', tz='UTC')


def _days(s):
    return (pd.to_datetime(s, utc=True) - pd.Timestamp('2020-01-01', tz='UTC')).dt.total_seconds().to_numpy() / 86400


def kalman(p, k=None):
    k = dict(KAL7, **(k or {}))
    q_t, q_l, v_t, v_l, w_int, side = k['q_t'], k['q_l'], k['v_t'], k['v_l'], k['w_int'], k['side']
    lam, kd = k['lam'], k['kd']
    th, tv, tt = {}, {}, {}
    lm, lv, lt = {}, {}, {}
    ts, te = _days(p.t_start), _days(p.t_end.fillna(pd.Timestamp('2100-01-01', tz='UTC')))
    y = p.blue_win.to_numpy(float)
    # soft result: lam * win + (1 - lam) * sigma(kd * g_area), so a lucky comeback counts less than a stomp (lam 1 = win/loss)
    ga = p.g_area.to_numpy(float) if 'g_area' in p else np.full(len(p), np.nan)
    ys = np.where(np.isfinite(ga), lam * y + (1 - lam) / (1 + np.exp(-kd * np.nan_to_num(ga))), y)
    B, R = p.blue_id.to_numpy(), p.red_id.to_numpy()
    BL, RL = p.b_league.to_numpy(), p.r_league.to_numpy()
    intl = p.intl.to_numpy(bool)

    def team(t, now):
        if t not in th:
            th[t], tv[t], tt[t] = 0.0, v_t, now
        return th[t], min(tv[t] + q_t * max(0.0, now - tt[t]), v_t)

    def league(l, now):
        if l not in lm:
            lm[l], lv[l], lt[l] = 0.0, v_l, now
        return lm[l], min(lv[l] + q_l * max(0.0, now - lt[l]), v_l)

    def state(i, now):
        (mb, vb), (mr, vr) = team(B[i], now), team(R[i], now)
        cross = BL[i] != RL[i]
        if cross:
            (ab, ub), (ar, ur) = league(BL[i], now), league(RL[i], now)
        else:
            ab = ar = ub = ur = 0.0
        return mb, vb, mr, vr, ab, ub, ar, ur, cross

    def apply(i):
        now = te[i]
        mb, vb, mr, vr, ab, ub, ar, ur, cross = state(i, now)
        z = mb - mr + ab - ar + side
        pr = 1 / (1 + math.exp(-z))
        w = w_int if intl[i] else 1.0
        V = vb + vr + ub + ur
        den = 1 + w * V * pr * (1 - pr)
        r = ys[i] - pr
        th[B[i]], tv[B[i]], tt[B[i]] = mb + w * vb * r / den, max(vb - w * vb * vb * pr * (1 - pr) / den, 1e-4), now
        th[R[i]], tv[R[i]], tt[R[i]] = mr - w * vr * r / den, max(vr - w * vr * vr * pr * (1 - pr) / den, 1e-4), now
        if cross:
            lm[BL[i]], lv[BL[i]], lt[BL[i]] = ab + w * ub * r / den, max(ub - w * ub * ub * pr * (1 - pr) / den, 1e-4), now
            lm[RL[i]], lv[RL[i]], lt[RL[i]] = ar - w * ur * r / den, max(ur - w * ur * ur * pr * (1 - pr) / den, 1e-4), now

    pend, out = [], np.zeros((len(p), 4))
    for i in range(len(p)):
        now = ts[i]
        while pend and pend[0][0] <= now:
            apply(heapq.heappop(pend)[1])
        mb, vb, mr, vr, ab, ub, ar, ur, cross = state(i, now)
        out[i] = (mb - mr, ab - ar, math.sqrt(ub + ur), mb - mr + ab - ar)
        if np.isfinite(y[i]):
            heapq.heappush(pend, (te[i], i))
    return pd.DataFrame(out, columns=['kt', 'kl', 'kl_sd', 'kp'], index=p.index)


def bt_online(p, since=None):
    """Weekly BT anchor + per-game Newton updates within the week. Returns (bt_weekly, bt7).
    since (a Timestamp): only weeks that end after it are computed (the tracker's live rows; earlier rows stay 0)."""
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
    y = p.blue_win.to_numpy(float)
    has = np.isfinite(y)
    t0 = p.t_start.min()
    tday = (p.t_start - t0).dt.total_seconds().to_numpy() / 86400
    eday = (p.t_end.fillna(pd.Timestamp('2100-01-01', tz='UTC')) - t0).dt.total_seconds().to_numpy() / 86400
    w_int = np.where(p.intl.to_numpy(bool), BT_INTL, 1.0)
    bt, bt7 = np.zeros(len(p)), np.zeros(len(p))
    s0 = -np.inf if since is None else (pd.Timestamp(since) - t0).total_seconds() / 86400
    for w0 in np.arange(0, tday.max() + 7, 7.0):
        cur = np.where((tday >= w0) & (tday < w0 + 7))[0]
        if not len(cur) or w0 + 7 <= s0:
            continue
        past = has & (eday < w0) & (eday >= w0 - 365)
        if past.sum() < 200 or len(set(y[past])) < 2:
            continue
        sw = w_int[past] * 0.5 ** ((w0 - eday[past]) / BT_HL)
        norm = sw.mean()
        m = LogisticRegression(C=BT_C, max_iter=3000).fit(X[past], y[past].astype(int), sample_weight=sw / norm)
        coef, b0 = m.coef_[0].copy(), float(m.intercept_[0])
        bt[cur] = X[cur] @ coef
        # games finished inside this week, in end order, update the anchor before the games that start after them
        wk = np.where(has & (eday >= w0) & (eday < w0 + 7))[0]
        if not len(wk):
            bt7[cur] = bt[cur]
            continue
        act = np.unique(np.r_[X[past].indices, X[cur].indices, X[wk].indices])
        pos = {c: j for j, c in enumerate(act)}
        Xa = X[past][:, act]
        pp = 1 / (1 + np.exp(-(Xa @ coef[act] + b0)))
        Wd = BT_C * (sw / norm) * pp * (1 - pp)
        H = np.eye(len(act)) + (Xa.T @ sp.diags(Wd) @ Xa).toarray()
        Hi = np.linalg.inv(H)
        th = coef[act].copy()
        order = wk[np.argsort(eday[wk])]
        ev = [(eday[j], 1, j) for j in order] + [(tday[j], 0, j) for j in cur]
        ev.sort(key=lambda e: (e[0], -e[1]))  # at equal times a game that ended counts before one that starts (strict: ended < start)
        for t, kind, j in ev:
            xr = X[j]
            idx = np.array([pos[c] for c in xr.indices])
            xv = xr.data
            if kind == 0:
                bt7[j] = float(xv @ th[idx])
                continue
            z = float(xv @ th[idx]) + b0
            pr = 1 / (1 + math.exp(-z))
            wj = BT_C * w_int[j] / norm
            hx = Hi[:, idx] @ xv
            a = wj * pr * (1 - pr)
            Hi -= np.outer(hx, hx) * (a / (1 + a * float(xv @ hx[idx])))
            th += (Hi[:, idx] @ xv) * (wj * (y[j] - pr))
    return bt, bt7


def surprise(p, kp):
    """day_dom / ser_dom / d2_dom from the per-game g_area and the Kalman logit kp (blue view)."""
    ga = p.g_area.to_numpy(float)
    z = np.asarray(kp, float)
    fit = np.isfinite(ga) & np.isfinite(p.blue_win.to_numpy(float)) & (p.t_start < E_END).to_numpy()
    b = np.polyfit(z[fit], ga[fit], 1) if fit.sum() > 100 else np.array([0.0, 0.0])
    s = np.where(np.isfinite(ga), ga - np.polyval(b, z), 0.0)
    ts, te = _days(p.t_start), _days(p.t_end.fillna(pd.Timestamp('2100-01-01', tz='UTC')))
    y = p.blue_win.to_numpy(float)
    B, R, M = p.blue_id.to_numpy(), p.red_id.to_numpy(), p.match_id.to_numpy()
    hist = defaultdict(list)
    pend, out = [], np.zeros((len(p), 3))
    for i in range(len(p)):
        now = ts[i]
        while pend and pend[0][0] <= now:
            t_, j = heapq.heappop(pend)
            hist[B[j]].append((t_, M[j], s[j]))
            hist[R[j]].append((t_, M[j], -s[j]))
        v = []
        for tm in (B[i], R[i]):
            day = ser = d2 = 0.0
            for t_, mid, x in reversed(hist[tm]):
                age = now - t_
                if age > 30:
                    break
                if age <= 0.5:
                    day += x
                if mid == M[i]:
                    ser += x
                d2 += x * 0.5 ** (age / 2.0)
            v.append((day, ser, d2))
        out[i] = np.subtract(v[0], v[1])
        if np.isfinite(y[i]):
            heapq.heappush(pend, (te[i], i))
    return pd.DataFrame(out, columns=['day_dom', 'ser_dom', 'd2_dom'], index=p.index), b


def gold_area(clock, gold):
    """mean over a 60 s grid of tanh(gold diff / 5k); clock in seconds from game start (any frame density)."""
    c, g = np.asarray(clock, float), np.asarray(gold, float)
    o = np.argsort(c)
    c, g = c[o], g[o]
    if len(c) < 3 or c[-1] < 300:
        return np.nan
    grid = np.arange(60.0, c[-1] + 1e-9, 60.0)
    return float(np.mean(np.tanh(np.interp(grid, c, g) / 5000)))


def gold_area_feed(d):
    """gold_area from a raw livestats frame table (ts, state, b_gold, r_gold): clock from the first frame with gold."""
    d = d[d.state.isin(['in_game', 'paused', 'finished'])]
    d = d.assign(_t=pd.to_datetime(d.ts, format='ISO8601')).sort_values('_t')
    d = d[(d.b_gold + d.r_gold) > 0]
    if len(d) < 3:
        return np.nan
    c = (d._t - d._t.iloc[0]).dt.total_seconds().to_numpy() + 65.0  # passive gold starts ~1:05 (live.features' anchor)
    return gold_area(c, (d.b_gold - d.r_gold).to_numpy(float))


FEATS = ['kt', 'kl', 'kl_sd', 'kp', 'bt7', 'day_dom', 'ser_dom', 'd2_dom']


def features(p, kal=None, since=None):
    """All v7 columns for table p (sorted by t_start); since: see bt_online."""
    p = p.reset_index(drop=True)
    K = kalman(p, kal)
    bt, bt7 = bt_online(p, since)
    S, b = surprise(p, K.kp.to_numpy())
    out = pd.concat([p[['game_id']], K, S], axis=1)
    out['bt_w'] = bt
    out['bt7'] = bt7
    return out
