"""Strictly causal pre-game features, one row per game, from the blue side's view.

A single pass over games ordered by start time. Before a game's features are
computed, every game that *ended* before it *started* is applied to the state,
so nothing from the game itself or anything concurrent leaks in.

Constructions (PLAN step 1 asks to compare them):
  elo_team_K   team Elo on esportsTeamId, K in {16, 32, 64}; new team starts at its league's mean
  elo_play_K   mean of the 5 players' Elo; each player is updated with the team result
  form_N       team win rate over its last N games (shrunk to 0.5 with 2 pseudo-games)
  h2h          blue wins minus red wins over earlier games between the two teams
  exp          mean log(1 + games played) of the 5 players
  roster       how many of the 5 players also played the team's previous game
  rest_d       days since the team's previous game
  series_diff  blue's game wins minus red's earlier in this series
"""
import heapq, math
from collections import defaultdict, deque
import numpy as np, pandas as pd

D = '../data'
KS = (16, 32, 64)


def e(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


class Elo:
    def __init__(self, k):
        self.k, self.r, self.league = k, {}, defaultdict(list)

    def get(self, key, league):
        if key not in self.r:
            lm = [self.r[x] for x in self.league[league][-200:] if x in self.r]
            self.r[key] = float(np.mean(lm)) if lm else 1500.0
            self.league[league].append(key)
        return self.r[key]


def build(t):
    t = t.sort_values('t_start').reset_index(drop=True)
    team = {k: Elo(k) for k in KS}
    play = {k: Elo(k) for k in KS}
    hist = defaultdict(lambda: deque(maxlen=20))  # team -> recent results
    last = {}  # team -> (t_end, players)
    h2h = defaultdict(int)  # (a, b) -> a's wins minus b's
    ngames = defaultdict(int)  # player -> games
    series = defaultdict(lambda: defaultdict(int))  # match -> team -> wins
    pending = []  # (t_end, idx)
    rows = []

    def apply(i):
        g = t.loc[i]
        y = g.blue_win
        b, r, lg = g.blue_id, g.red_id, g.league
        for k in KS:
            rb, rr = team[k].get(b, lg), team[k].get(r, lg)
            d = k * (y - e(rb, rr))
            team[k].r[b] += d; team[k].r[r] -= d
            pb = [play[k].get(p, lg) for p in g.b_players]
            pr = [play[k].get(p, lg) for p in g.r_players]
            d = k * (y - e(np.mean(pb), np.mean(pr)))
            for p in g.b_players: play[k].r[p] += d
            for p in g.r_players: play[k].r[p] -= d
        hist[b].append(y); hist[r].append(1 - y)
        h2h[(b, r)] += 1 if y else -1; h2h[(r, b)] -= 1 if y else -1
        for p in list(g.b_players) + list(g.r_players):
            ngames[p] += 1
        last[b] = (g.t_end, set(g.b_players)); last[r] = (g.t_end, set(g.r_players))
        series[g.match_id][b if y else r] += 1

    for i, g in t.iterrows():
        while pending and pending[0][0] <= g.t_start:
            apply(heapq.heappop(pending)[1])
        b, r, lg = g.blue_id, g.red_id, g.league
        f = dict(game_id=g.game_id)
        for k in KS:
            f[f'elo_team_{k}'] = team[k].get(b, lg) - team[k].get(r, lg)
            f[f'elo_play_{k}'] = np.mean([play[k].get(p, lg) for p in g.b_players]) - \
                np.mean([play[k].get(p, lg) for p in g.r_players])
        for n in (5, 10, 20):
            fb = (sum(list(hist[b])[-n:]) + 1) / (len(list(hist[b])[-n:]) + 2)
            fr = (sum(list(hist[r])[-n:]) + 1) / (len(list(hist[r])[-n:]) + 2)
            f[f'form_{n}'] = fb - fr
        f['h2h'] = h2h[(b, r)]
        f['exp'] = np.mean([math.log1p(ngames[p]) for p in g.b_players]) - \
            np.mean([math.log1p(ngames[p]) for p in g.r_players])
        f['roster'] = (len(last[b][1] & set(g.b_players)) if b in last else 0) - \
            (len(last[r][1] & set(g.r_players)) if r in last else 0)
        rb = (g.t_start - last[b][0]).total_seconds() / 86400 if b in last else 30
        rr = (g.t_start - last[r][0]).total_seconds() / 86400 if r in last else 30
        f['rest_d'] = np.clip(rb, 0, 30) - np.clip(rr, 0, 30)
        f['series_diff'] = series[g.match_id][b] - series[g.match_id][r]
        f['n_min_games'] = min(len(hist[b]), len(hist[r]))
        rows.append(f)
        heapq.heappush(pending, (g.t_end, i))
    return t.merge(pd.DataFrame(rows), on='game_id')


if __name__ == '__main__':
    t = pd.read_parquet(f'{D}/game_table.parquet')
    t = t[t.b_players.map(lambda l: all(l)) & t.r_players.map(lambda l: all(l))]
    p = build(t)
    p.to_parquet(f'{D}/pregame.parquet')
    print(p.shape)
