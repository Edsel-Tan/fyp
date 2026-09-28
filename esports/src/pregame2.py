"""Directions 3-5: extra strictly causal pre-game features, one row per game (blue minus red unless noted).

Same pass as pregame.py: a game's outcome enters the state only once it has ended before the current game starts.
Counts decay with a half-life of HL days, so old patches and old rosters fade.

Draft / patch (direction 5)
  champ_all      sum over roles of logit of the champion-role's decayed win rate (shrunk to 0.5, 10 pseudo-games)
  champ_patch    the same on the current patch only, shrunk to champ_all's rate (20 pseudo-games)
  lane_mu        champion-vs-champion lane matchup: shrunk mean residual (y - expectation from champ_patch)
  pc_exp         player-on-champion experience: mean log(1 + decayed games)
  pc_wr          player-on-champion win-rate residual vs champ_patch, shrunk (8 pseudo-games)
  scaling        sum of champion late-game lean: logit wr in games >= 33 min minus logit wr in shorter games (shrunk)
  patch_age      days since the patch was first seen (not a difference)
  side_patch     blue-side log-odds on the current patch, shrunk to the long-run side rate (not a difference)
Team strength (direction 4)
  role_elo_k     win-based player Elo (K=32), role k blue player minus red player (k = top, jgl, mid, bot, sup)
  lane_elo_k     lane Elo from gold difference at 15:00 vs the lane opponent (score = sigmoid(gd15 / 1000), K=32)
  conv           team conversion: shrunk mean of (y - q_gd15) over past games, q_gd15 = burn-in logistic on gold at 15
  share_b_k/r_k  team's decayed share of team gold by role (carry dependence; per side, not a difference)
  league_elo     league-strength Elo updated only on inter-league games
  hinge_b_k/r_k  player leverage: shrunk mean of (team y - q_gd15) * tanh(own lane gd15 / 1000) over the player's past games;
                 > 0 means the team over-converts when this player is ahead (per side, for in-game gold x hinge terms)
Head to head (direction 3)
  h2h_res        team h2h: shrunk mean residual y - E_elo over earlier meetings (5 pseudo-games)
  pp_res         player pairs (lane opponents): shrunk mean of y - E_elo over their earlier meetings, summed over roles
  lane_h2h       lane-opponent pairs: shrunk mean lane-score residual vs lane Elo expectation, summed over roles
"""
import heapq, math, glob, os
from collections import defaultdict
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

D = '../data'
HL = 120.0  # half-life, days
ROLES = ['top', 'jungle', 'mid', 'bottom', 'support']


def lg(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def sig(x):
    return 1 / (1 + math.exp(-x))


def e(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


class Dec:
    """Decayed (sum, count) per key."""
    def __init__(self):
        self.s, self.n, self.t = defaultdict(float), defaultdict(float), {}

    def _f(self, k, t):
        return 0.5 ** ((t - self.t[k]) / HL) if k in self.t else 1.0

    def add(self, k, x, t, w=1.0):
        f = self._f(k, t)
        self.s[k] = self.s[k] * f + x * w
        self.n[k] = self.n[k] * f + w
        self.t[k] = t

    def get(self, k, t):
        if k not in self.t:
            return 0.0, 0.0
        f = self._f(k, t)
        return self.s[k] * f, self.n[k] * f


def role_gold(gid):
    """Per-role gold at ~15:00 (blue minus red) and each player's share of team gold at the last frame."""
    try:
        d = pd.read_parquet(f'{D}/frames60/{gid}.parquet')
    except Exception:
        return None
    d = d[d.state.isin(['in_game', 'paused', 'finished']) & ((d.b_gold + d.r_gold) > 0)]
    if len(d) < 3:
        return None
    t = pd.to_datetime(d.ts, format='ISO8601')
    el = (t - t.iloc[0]).dt.total_seconds().to_numpy() + 90  # clock approx (first gold frame ~ 1:30)
    out = dict(game_id=gid)
    i = int(np.argmin(np.abs(el - 900)))
    ok = abs(el[i] - 900) < 60
    last = d.iloc[-1]
    bt = sum(last[f'p{k}_totalGold'] for k in range(1, 6)) or 1
    rt = sum(last[f'p{k}_totalGold'] for k in range(6, 11)) or 1
    for k in range(5):
        out[f'gd15_{k}'] = float(d[f'p{k + 1}_totalGold'].iloc[i] - d[f'p{k + 6}_totalGold'].iloc[i]) if ok else np.nan
        out[f'bsh_{k}'] = last[f'p{k + 1}_totalGold'] / bt
        out[f'rsh_{k}'] = last[f'p{k + 6}_totalGold'] / rt
    return out


def build(t, rg, q15):
    t = t.sort_values('t_start').reset_index(drop=True)
    rg = rg.set_index('game_id')
    days = lambda ts: ts.value / 86400e9
    ch_all, ch_patch, lane_mu, pc, pcw, scal_l, scal_s = Dec(), Dec(), Dec(), Dec(), Dec(), Dec(), Dec()
    side_all, side_patch = Dec(), Dec()
    role_elo, lane_elo = defaultdict(lambda: 1500.0), defaultdict(lambda: 1500.0)
    team_elo = defaultdict(lambda: 1500.0)
    conv, h2h, pp, lh2h, sh, hin = Dec(), Dec(), Dec(), Dec(), Dec(), Dec()
    lelo = defaultdict(lambda: 1500.0)
    patch_first = {}
    pending, rows = [], []

    def cp(ck, pv, now):
        sa, na = ch_all.get(ck, now)
        pa = (sa + 5) / (na + 10)
        sp, npch = ch_patch.get((pv,) + ck, now)
        return pa, (sp + 20 * pa) / (npch + 20)

    def apply(i):
        g = t.loc[i]
        y, now, pv = float(g.blue_win), days(g.t_end), g.pv
        long_ = g.dur_s >= 33 * 60
        eb = e(team_elo[g.blue_id], team_elo[g.red_id])
        side_all.add('s', y, now); side_patch.add(pv, y, now)
        for k, role in enumerate(ROLES):
            cb, cr = (role, g.b_champs[k]), (role, g.r_champs[k])
            pb_, pr_ = cp(cb, pv, now)[1], cp(cr, pv, now)[1]
            ch_all.add(cb, y, now); ch_all.add(cr, 1 - y, now)
            ch_patch.add((pv,) + cb, y, now); ch_patch.add((pv,) + cr, 1 - y, now)
            ex = sig(lg(pb_) - lg(pr_))
            lane_mu.add((role, g.b_champs[k], g.r_champs[k]), y - ex, now)
            lane_mu.add((role, g.r_champs[k], g.b_champs[k]), ex - y, now)
            pl_b, pl_r = g.b_players[k], g.r_players[k]
            pc.add((pl_b, g.b_champs[k]), 1, now); pc.add((pl_r, g.r_champs[k]), 1, now)
            pcw.add((pl_b, g.b_champs[k]), y - pb_, now); pcw.add((pl_r, g.r_champs[k]), (1 - y) - pr_, now)
            (scal_l if long_ else scal_s).add(g.b_champs[k], y, now)
            (scal_l if long_ else scal_s).add(g.r_champs[k], 1 - y, now)
            # role (win) Elo, per player
            ee = e(role_elo[pl_b], role_elo[pl_r])
            role_elo[pl_b] += 32 * (y - ee); role_elo[pl_r] -= 32 * (y - ee)
            gd = rg.at[g.game_id, f'gd15_{k}'] if g.game_id in rg.index else np.nan
            if not np.isnan(gd):
                s = sig(gd / 1000)
                el = e(lane_elo[pl_b], lane_elo[pl_r])
                lane_elo[pl_b] += 32 * (s - el); lane_elo[pl_r] -= 32 * (s - el)
                key = (role,) + tuple(sorted((pl_b, pl_r)))
                lh2h.add(key, (s - el) * (1 if pl_b < pl_r else -1), now)
            key = (role,) + tuple(sorted((pl_b, pl_r)))
            pp.add(key, (y - eb) * (1 if pl_b < pl_r else -1), now)
        if g.game_id in rg.index:
            r = rg.loc[g.game_id]
            for k in range(5):
                sh.add((g.blue_id, k), r[f'bsh_{k}'], now); sh.add((g.red_id, k), r[f'rsh_{k}'], now)
            if not np.isnan(r.gd15_0):
                qq = q15(sum(r[f'gd15_{k}'] for k in range(5)))
                conv.add(g.blue_id, y - qq, now); conv.add(g.red_id, (1 - y) - (1 - qq), now)
                for k in range(5):
                    lead = math.tanh(r[f'gd15_{k}'] / 1000)
                    hin.add(g.b_players[k], (y - qq) * lead, now); hin.add(g.r_players[k], ((1 - y) - (1 - qq)) * -lead, now)
        key = tuple(sorted((g.blue_id, g.red_id)))
        h2h.add(key, (y - eb) * (1 if g.blue_id < g.red_id else -1), now)
        team_elo[g.blue_id] += 32 * (y - eb); team_elo[g.red_id] -= 32 * (y - eb)
        if g.b_league != g.r_league:
            el = e(lelo[g.b_league], lelo[g.r_league])
            lelo[g.b_league] += 16 * (y - el); lelo[g.r_league] -= 16 * (y - el)

    for i, g in t.iterrows():
        while pending and pending[0][0] <= g.t_start:
            apply(heapq.heappop(pending)[1])
        now, pv = days(g.t_start), g.pv
        patch_first.setdefault(pv, now)
        f = dict(game_id=g.game_id, patch_age=now - patch_first[pv])
        sa, na = side_all.get('s', now)
        ps = (sa + 10) / (na + 20)
        sp, npch = side_patch.get(pv, now)
        f['side_patch'] = lg((sp + 50 * ps) / (npch + 50))
        ca = cpt = lm = pe = pw = sc = 0.0
        for k, role in enumerate(ROLES):
            for sgn, ch, pl, op in ((1, g.b_champs[k], g.b_players[k], g.r_champs[k]), (-1, g.r_champs[k], g.r_players[k], g.b_champs[k])):
                pa, pp_ = cp((role, ch), pv, now)
                ca += sgn * lg(pa); cpt += sgn * lg(pp_)
                s_, n_ = lane_mu.get((role, ch, op), now)
                lm += sgn * s_ / (n_ + 10)
                pe += sgn * math.log1p(pc.get((pl, ch), now)[1])
                s_, n_ = pcw.get((pl, ch), now)
                pw += sgn * s_ / (n_ + 8)
                sl, nl = scal_l.get(ch, now); ss, ns = scal_s.get(ch, now)
                sc += sgn * (lg((sl + 5) / (nl + 10)) - lg((ss + 5) / (ns + 10)))
        f.update(champ_all=ca, champ_patch=cpt, lane_mu=lm / 2, pc_exp=pe / 5, pc_wr=pw, scaling=sc)
        ppr = lhr = 0.0
        for k, role in enumerate(ROLES):
            pb, pr = g.b_players[k], g.r_players[k]
            f[f'role_elo_{k}'] = role_elo[pb] - role_elo[pr]
            f[f'lane_elo_{k}'] = lane_elo[pb] - lane_elo[pr]
            key = (role,) + tuple(sorted((pb, pr)))
            sgn = 1 if pb < pr else -1
            s_, n_ = pp.get(key, now); ppr += sgn * s_ / (n_ + 5)
            s_, n_ = lh2h.get(key, now); lhr += sgn * s_ / (n_ + 5)
            for side, pl in (('b', pb), ('r', pr)):
                s_, n_ = hin.get(pl, now)
                f[f'hinge_{side}_{k}'] = s_ / (n_ + 20)
            for side, tid in (('b', g.blue_id), ('r', g.red_id)):
                s_, n_ = sh.get((tid, k), now)
                f[f'share_{side}_{k}'] = (s_ + 0.2 * 3) / (n_ + 3)
        f['pp_res'], f['lane_h2h'] = ppr, lhr
        s_, n_ = conv.get(g.blue_id, now); s2, n2 = conv.get(g.red_id, now)
        f['conv'] = s_ / (n_ + 10) - s2 / (n2 + 10)
        key = tuple(sorted((g.blue_id, g.red_id)))
        s_, n_ = h2h.get(key, now)
        f['h2h_res'] = (1 if g.blue_id < g.red_id else -1) * s_ / (n_ + 5)
        f['league_elo'] = lelo[g.b_league] - lelo[g.r_league]
        rows.append(f)
        heapq.heappush(pending, (g.t_end, i))
    return pd.DataFrame(rows)


def main():
    p = pd.read_parquet(f'{D}/pregame.parquet')
    p['pv'] = p.patch.str.split('.').str[:2].str.join('.')
    # each team's home league: its most common non-international league so far (static mode is fine: leagues rarely change)
    from eval_pregame import INTL
    tl = pd.concat([p[['blue_id', 'league']].rename(columns={'blue_id': 'tid'}), p[['red_id', 'league']].rename(columns={'red_id': 'tid'})])
    home = tl[~tl.league.isin(INTL)].groupby('tid').league.agg(lambda s: s.value_counts().index[0])
    p['b_league'] = p.blue_id.map(home).fillna(p.league)
    p['r_league'] = p.red_id.map(home).fillna(p.league)
    cache = f'{D}/role_gold.parquet'
    if os.path.exists(cache):
        rg = pd.read_parquet(cache)
    else:
        with ProcessPoolExecutor(24) as ex:
            rg = pd.DataFrame([x for x in ex.map(role_gold, p.game_id, chunksize=64) if x is not None])
        rg.to_parquet(cache)
    # q_gd15: logistic on total gold diff at 15 min, fit on burn-in games only (before 2024-04)
    from sklearn.linear_model import LogisticRegression
    b = p[p.t_start < pd.Timestamp('2024-04-01', tz='UTC')].merge(rg, on='game_id').dropna(subset=['gd15_0'])
    x = b[[f'gd15_{k}' for k in range(5)]].sum(1).to_numpy()[:, None] / 1000
    m = LogisticRegression().fit(x, b.blue_win.astype(int))
    a0, a1 = m.intercept_[0], m.coef_[0, 0]
    print(f'q_gd15 burn-in fit on {len(b)} games: logit = {a0:.3f} + {a1:.3f} * gd15/1000')
    q15 = lambda gd: sig(a0 + a1 * gd / 1000)
    f = build(p, rg, q15)
    f.to_parquet(f'{D}/pregame2.parquet')
    print(f.shape)
    print(f.describe().T[['mean', 'std']].round(3).to_string())


if __name__ == '__main__':
    main()
