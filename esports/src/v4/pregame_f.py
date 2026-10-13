"""v4 step 2: pre-game group F from PROGRESS "Missing inputs" (2026-10-05), one row per game, strictly causal.

Series state (from earlier games of the same match, sides from livestats blue_id / red_id):
  map_no        game number in the series (not a difference; enters through the interactions below)
  mp            (blue on match point) - (red on match point)
  decider       1 if both teams are on match point (the deciding map), else 0
  dec_elo       decider x elo_play_32 / 400 (does the stronger team convert deciders differently?)
  prev_loser    +1 if the previous game's loser is on blue now, -1 if on red, 0 in game 1 (most rules let the loser choose side)
Side-specific form (Guan & Tung's same-side form):
  ss_res        blue team's decayed mean (y - E_elo) on blue side minus red team's on red side, shrunk with 10 pseudo-games (HL 120 d)
  ss_ewm        blue team's EWMA (alpha 0.1) of blue-side results minus red team's of red-side results (both start at 0.5)
Fearless draft (champions used by a team earlier in the series are unavailable; every league from 2025, NACL in 2024):
  fearless      1 if the (league, year) never repeats a team champion within a series (detected from the table; rules are pre-season)
  pool          mean over roles of log(1 + champions with decayed count >= 1) per player, blue minus red (HL 120 d)
  pool_dep      pool x (map_no - 1) x fearless
  dep           fearless x champion comfort already spent in this series: sum over the team's earlier games of the mean
                log(1 + decayed player-on-champion games) of the champions it used, blue minus red
Market:
  pm_pre        Polymarket per-game market P(blue) before the game: volume-weighted price of the last 30 min of fills before
                t_start (falls back to the last fill within 6 h); NaN when the game has no linked market
Writes data/v4/pregame_f.parquet.
"""
import heapq, math, os, sys
from collections import defaultdict
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
HL = 120.0


class Dec:
    """Decayed (sum, count) per key."""
    def __init__(self):
        self.s, self.n, self.t = defaultdict(float), defaultdict(float), {}

    def _f(self, k, t):
        return 0.5 ** ((t - self.t[k]) / HL) if k in self.t else 1.0

    def add(self, k, x, t):
        f = self._f(k, t)
        self.s[k] = self.s[k] * f + x
        self.n[k] = self.n[k] * f + 1
        self.t[k] = t

    def get(self, k, t):
        f = self._f(k, t)
        return self.s[k] * f, self.n[k] * f


def fearless_table(t):
    rows = []
    for m, g in t.sort_values('number').groupby('match_id'):
        if len(g) < 2:
            continue
        for tid in set(g.blue_id) | set(g.red_id):
            ch = [c for _, r in g.iterrows() for c in (r.b_champs if r.blue_id == tid else r.r_champs if r.red_id == tid else [])]
            rows.append(dict(league=g.league.iloc[0], yr=g.t_start.iloc[0].year, rep=len(ch) != len(set(ch))))
    r = pd.DataFrame(rows).groupby(['league', 'yr']).rep.mean()
    return {k for k, v in r.items() if v < 0.05}


def pm_pre(t):
    mk = pd.read_parquet(f'{D}/pm_game_markets.parquet').sort_values('vol', ascending=False).drop_duplicates('game_id')
    fl = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event', 'usd'])
    fl = fl[fl.condition_id.isin(mk.condition_id)].merge(mk[['condition_id', 'game_id', 'seq1_blue']], on='condition_id')
    fl['p_blue'] = np.where(fl.seq1_blue == 1, fl.p_event, 1 - fl.p_event)
    fl['w'] = fl.usd.abs()
    fl['t0'] = fl.game_id.map(t.set_index('game_id').t_start)
    fl['ts'] = pd.to_datetime(fl.ts, unit='s', utc=True)
    pre = fl[(fl.ts < fl.t0) & (fl.ts >= fl.t0 - pd.Timedelta('6h'))]
    out = {}
    for gid, g in pre.groupby('game_id'):
        last = g[g.ts >= g.t0.iloc[0] - pd.Timedelta('30min')]
        if len(last):
            out[gid] = float(np.average(last.p_blue, weights=last.w.clip(lower=1e-9)))
        else:
            out[gid] = float(g.sort_values('ts').p_blue.iloc[-1])
    return pd.Series(out, name='pm_pre')


def build(t, elo):
    t = t.sort_values('t_start').reset_index(drop=True)
    fear = fearless_table(t)
    pending = []
    side_res, side_ewm = Dec(), {}
    pcn = Dec()  # (player, champ) decayed games
    pchamps = defaultdict(set)  # player -> champions seen (for the pool; decayed counts in pcn)
    series = defaultdict(dict)  # match -> number -> (blue_id, red_id, blue_win, b_spent, r_spent)
    rows = []

    def day(ts):
        return ts.value / 86400e9

    def apply(i):
        g = t.loc[i]
        y, d0 = float(g.blue_win), day(g.t_end)
        E = 1 / (1 + 10 ** (-elo.get(g.game_id, 0.0) / 400))
        side_res.add((g.blue_id, 'b'), y - E, d0)
        side_res.add((g.red_id, 'r'), (1 - y) - (1 - E), d0)
        side_ewm[(g.blue_id, 'b')] = 0.9 * side_ewm.get((g.blue_id, 'b'), 0.5) + 0.1 * y
        side_ewm[(g.red_id, 'r')] = 0.9 * side_ewm.get((g.red_id, 'r'), 0.5) + 0.1 * (1 - y)
        for ps, cs in ((g.b_players, g.b_champs), (g.r_players, g.r_champs)):
            for p, c in zip(ps, cs):
                pcn.add((p, c), 1.0, d0)
                pchamps[p].add(c)
        series[g.match_id][g.number]['y'] = y

    def pool(ps, d0):
        return np.mean([math.log1p(sum(pcn.get((p, c), d0)[1] >= 1 for c in pchamps[p])) for p in ps])

    def comfort(ps, cs, d0):
        return np.mean([math.log1p(pcn.get((p, c), d0)[1]) for p, c in zip(ps, cs)])

    for i, g in t.iterrows():
        while pending and pending[0][0] <= g.t_start:
            apply(heapq.heappop(pending)[1])
        d0 = day(g.t_start)
        b, r, bo, n = g.blue_id, g.red_id, int(g.bo), int(g.number)
        fl = int((g.league, g.t_start.year) in fear)
        # series state from earlier games whose result is known (they ended before this one started)
        earlier = {k: v for k, v in series[g.match_id].items() if k < n and 'y' in v}
        wins = defaultdict(int)
        spent = defaultdict(float)
        for k, v in earlier.items():
            wins[v['blue'] if v['y'] == 1 else v['red']] += 1
            spent[v['blue']] += v['b_comf']
            spent[v['red']] += v['r_comf']
        need = bo // 2 + bo % 2  # wins to take the series
        mpb, mpr = int(bo > 1 and wins[b] == need - 1), int(bo > 1 and wins[r] == need - 1)
        prev = earlier.get(n - 1)
        if prev is None:
            pl = 0
        else:
            loser = prev['red'] if prev['y'] == 1 else prev['blue']
            pl = 1 if loser == b else (-1 if loser == r else 0)
        sb, nb_ = side_res.get((b, 'b'), d0)
        sr, nr_ = side_res.get((r, 'r'), d0)
        pb, pr_ = pool(g.b_players, d0), pool(g.r_players, d0)
        e32 = elo.get(g.game_id, 0.0)
        rows.append(dict(game_id=g.game_id, map_no=n, mp=mpb - mpr, decider=mpb * mpr, dec_elo=mpb * mpr * e32 / 400,
                         prev_loser=pl, ss_res=sb / (nb_ + 10) - sr / (nr_ + 10),
                         ss_ewm=side_ewm.get((b, 'b'), 0.5) - side_ewm.get((r, 'r'), 0.5),
                         fearless=fl, pool=pb - pr_, pool_dep=(pb - pr_) * (n - 1) * fl, dep=fl * (spent[b] - spent[r])))
        series[g.match_id][n] = dict(blue=b, red=r, b_comf=comfort(g.b_players, g.b_champs, d0),
                                     r_comf=comfort(g.r_players, g.r_champs, d0))
        heapq.heappush(pending, (g.t_end, i))
    return pd.DataFrame(rows)


def main():
    p = pd.read_parquet(f'{D}/pregame.parquet')
    elo = p.set_index('game_id').elo_team_32.to_dict()
    t = pd.read_parquet(f'{D}/game_table.parquet')
    t = t[t.game_id.isin(p.game_id)]
    f = build(t, elo)
    f = f.merge(pm_pre(t).rename_axis('game_id').reset_index(), on='game_id', how='left')
    os.makedirs(f'{D}/v4', exist_ok=True)
    f.to_parquet(f'{D}/v4/pregame_f.parquet')
    pd.set_option('display.width', 250)
    print(f.shape)
    print(f.drop(columns='game_id').describe().round(3).T.to_string())
    print('pm_pre available:', f.pm_pre.notna().sum())


if __name__ == '__main__':
    main()
