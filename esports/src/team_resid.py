"""Team strength net of the roster: is team Elo stale after roster changes, and does a roster-adjusted team term help?

pregame.py's elo_team_K is keyed on esportsTeamId and never resets, so after a roster change it still carries the old
roster's results. Two fixes, both strictly causal (same pending-heap pass as pregame.py), blue minus red:

  res_lin        literal version: at each game, team Elo T and the roster's mean player Elo P (both K=32, both as of that game)
                 for every side in the trailing 365 days; OLS T = a + b P refit every 250 games; feature = residual T - a - b P.
                 Note: with elo_team and elo_play already in a linear model this is (almost) a reparameterisation, and T is
                 still stale after a roster swap, so it should add little.
  hier_p/hier_u  hierarchical Elo: strength = mean player rating + team residual u. One surprise d = y - E(strength) updates
                 every player by Kp*d and u by Ku*d, so u only learns what the players' ratings don't explain. Before each
                 use, u is shrunk by (players kept / 5)^gamma when the roster differs from the team's last game, and decays
                 toward 0 with a half-life of HL days. A grid of (Ku, gamma, HL) is written; eval picks one on validation.
  ov             min over the two sides of the players shared with the team's game ~20 games back (for breakdowns; NaN early)

Coach: the slot is strength += c[coach], updated like u. No coach-by-date source is available here (lolesports getTeams
has players only; Leaguepedia is rate-limited), so it is not built.
"""
import heapq, itertools
from collections import defaultdict
import numpy as np, pandas as pd
from pregame import Elo, e

D = '../data'
KP = 32
GRID = list(itertools.product((8, 16, 32), (0, 1, 2), (np.inf, 180.0)))  # (Ku, gamma, HL)


def cname(ku, g, hl):
    return f'{ku}_{g}_{"inf" if np.isinf(hl) else int(hl)}'


def build_lin(t):
    team, play = Elo(KP), Elo(KP)
    snap = []  # (t_start_days, T, P) per side, pre-game
    coef, pending, rows = (0.0, 1.0), [], []
    hist = defaultdict(list)
    days = lambda ts: ts.value / 86400e9

    def apply(i):
        g = t.loc[i]
        y, b, r, lg = g.blue_win, g.blue_id, g.red_id, g.league
        rb, rr = team.get(b, lg), team.get(r, lg)
        d = KP * (y - e(rb, rr)); team.r[b] += d; team.r[r] -= d
        pb = np.mean([play.get(x, lg) for x in g.b_players]); pr = np.mean([play.get(x, lg) for x in g.r_players])
        d = KP * (y - e(pb, pr))
        for x in g.b_players: play.r[x] += d
        for x in g.r_players: play.r[x] -= d

    for i, g in t.iterrows():
        while pending and pending[0][0] <= g.t_start:
            apply(heapq.heappop(pending)[1])
        now, lg = days(g.t_start), g.league
        if i % 250 == 0 and len(snap) > 500:
            s = np.array([x for x in snap if x[0] >= now - 365])
            coef = tuple(np.polyfit(s[:, 2], s[:, 1], 1)[::-1])
        f = dict(game_id=g.game_id)
        res, ov = [], []
        for tid, pls in ((g.blue_id, g.b_players), (g.red_id, g.r_players)):
            T, P = team.get(tid, lg), np.mean([play.get(x, lg) for x in pls])
            snap.append((now, T, P))
            res.append(T - coef[0] - coef[1] * P)
            h = hist[tid]
            ov.append(len(set(h[-20]) & set(pls)) if len(h) >= 20 else np.nan)
            h.append(list(pls))
        f['res_lin'] = res[0] - res[1]
        f['lin_b'] = coef[1]
        f['ov'] = min(ov)
        rows.append(f)
        heapq.heappush(pending, (g.t_end, i))
    return pd.DataFrame(rows)


def build_hier(t, ku, gamma, hl):
    play = Elo(KP)
    u, last = defaultdict(float), {}  # team -> residual; team -> (roster set, time)
    pending, rows = [], []
    days = lambda ts: ts.value / 86400e9

    def eff(tid, pls, now):
        if tid not in last:
            return 0.0
        ros, t0 = last[tid]
        x = u[tid] * (len(ros & set(pls)) / 5) ** gamma
        return x * 0.5 ** ((now - t0) / hl) if np.isfinite(hl) else x

    def apply(i):
        g = t.loc[i]
        y, now, lg = g.blue_win, days(g.t_end), g.league
        ub, ur = eff(g.blue_id, g.b_players, now), eff(g.red_id, g.r_players, now)
        pb = np.mean([play.get(x, lg) for x in g.b_players]); pr = np.mean([play.get(x, lg) for x in g.r_players])
        d = y - e(pb + ub, pr + ur)
        for x in g.b_players: play.r[x] += KP * d
        for x in g.r_players: play.r[x] -= KP * d
        u[g.blue_id] = ub + ku * d; u[g.red_id] = ur - ku * d
        last[g.blue_id] = (set(g.b_players), now); last[g.red_id] = (set(g.r_players), now)

    for i, g in t.iterrows():
        while pending and pending[0][0] <= g.t_start:
            apply(heapq.heappop(pending)[1])
        now, lg = days(g.t_start), g.league
        pb = np.mean([play.get(x, lg) for x in g.b_players]); pr = np.mean([play.get(x, lg) for x in g.r_players])
        c = cname(ku, gamma, hl)
        rows.append({'game_id': g.game_id, f'hier_p_{c}': pb - pr,
                     f'hier_u_{c}': eff(g.blue_id, g.b_players, now) - eff(g.red_id, g.r_players, now)})
        heapq.heappush(pending, (g.t_end, i))
    return pd.DataFrame(rows).set_index('game_id')


def _hier(args):
    return build_hier(_T, *args)


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    _T = pd.read_parquet(f'{D}/pregame.parquet')[['game_id', 't_start', 't_end', 'league', 'blue_id', 'red_id',
                                                   'b_players', 'r_players', 'blue_win']]
    _T = _T.sort_values('t_start').reset_index(drop=True)
    with ProcessPoolExecutor(len(GRID) + 1) as ex:
        lin = ex.submit(build_lin, _T)
        H = list(ex.map(_hier, GRID))
    out = lin.result().set_index('game_id').join(H)
    out.reset_index().to_parquet(f'{D}/team_resid.parquet')
    print(out.shape)
    print(out[['res_lin', 'lin_b', 'ov']].describe().round(3).to_string())
