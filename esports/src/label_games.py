"""Build the game table: one row per game with frames, sides from livestats,
end-of-game features for winner inference, and every available ground truth.

Ground truths (none is a per-game winner field; lolesports has none):
  sweep  - series won 2-0 / 3-0 / 1-0: every game went to the series winner
  final  - the last game of a completed series went to the series winner
  pm     - the per-game Polymarket market's resolution (linked at score >= 85)
"""
import glob, json, os, sys
import numpy as np, pandas as pd

D = '../data'
m = pd.read_parquet(f'{D}/matches.parquet')
g = pd.read_parquet(f'{D}/games.parquet')
g = g[g.gstate == 'completed']


def dead(row, side):
    ids = range(1, 6) if side == 'b' else range(6, 11)
    return sum(row[f'p{i}_currentHealth'] == 0 for i in ids)


def features(gid):
    f = f'{D}/frames60/{gid}.parquet'
    d = pd.read_parquet(f)
    meta = json.load(open(f.replace('.parquet', '.meta.json')))
    d = d[d.state.isin(['in_game', 'finished', 'paused'])]
    if len(d) < 3 or any(f'p{i}_currentHealth' not in d for i in range(1, 11)):
        return None  # fewer than 10 participants in the feed
    ts = pd.to_datetime(d.ts, format='ISO8601')
    last = d.iloc[-1]
    r = dict(game_id=str(gid), n_frames=len(d), finished=(d.state == 'finished').any(),
             t_start=ts.iloc[0], t_end=ts.iloc[-1], dur_s=(ts.iloc[-1] - ts.iloc[0]).total_seconds(),
             blue_id=meta['blueTeamMetadata']['esportsTeamId'], red_id=meta['redTeamMetadata']['esportsTeamId'],
             patch=meta.get('patchVersion'))
    for sd, key in (('b', 'blueTeamMetadata'), ('r', 'redTeamMetadata')):
        pm = sorted(meta[key]['participantMetadata'], key=lambda p: p['participantId'])
        r[f'{sd}_players'] = [p.get('esportsPlayerId') for p in pm]
        r[f'{sd}_champs'] = [p.get('championId') for p in pm]
        r[f'{sd}_roles'] = [p.get('role') for p in pm]
    for k in ('towers', 'inhibs', 'gold', 'kills', 'barons'):
        r[f'e_{k}'] = last[f'b_{k}'] - last[f'r_{k}']
    r['e_b_towers'], r['e_r_towers'] = last.b_towers, last.r_towers
    r['e_b_inhibs'], r['e_r_inhibs'] = last.b_inhibs, last.r_inhibs
    r['e_dead'] = dead(last, 'r') - dead(last, 'b')  # >0: more red players dead at the end
    for w in (60, 120, 180):
        prev = d[ts <= ts.iloc[-1] - pd.Timedelta(f'{w}s')]
        prev = prev.iloc[-1] if len(prev) else d.iloc[0]
        r[f'push{w}'] = int((last.b_towers - prev.b_towers) + (last.b_inhibs - prev.b_inhibs)
                            - (last.r_towers - prev.r_towers) - (last.r_inhibs - prev.r_inhibs))
        r[f'kills{w}'] = int((last.b_kills - prev.b_kills) - (last.r_kills - prev.r_kills))
    return r


def old_rule(r):
    """The rule in winner.py, from the table's columns."""
    push = r.push120
    end = 4 * r.e_inhibs + r.e_towers + r.e_gold / 3000
    return 10 * np.sign(push) * min(abs(push), 3) + np.clip(end, -9, 9) / 10


def main():
    have = {os.path.basename(p)[:-8] for p in glob.glob(f'{D}/frames60/*.parquet')}
    rows = [x for gid in g.game_id if gid in have and (x := features(gid))]
    t = pd.DataFrame(rows).merge(g[['game_id', 'match_id', 'number', 'blue_team_id', 'red_team_id', 'blue_code', 'red_code']],
                                 on='game_id')
    # team id -> code, from either slot of games.parquet (sides there are random, codes are not)
    code = dict(zip(g.blue_team_id, g.blue_code)) | dict(zip(g.red_team_id, g.red_code))
    t['blue_code_ls'] = t.blue_id.map(code)
    t['red_code_ls'] = t.red_id.map(code)
    t = t.merge(m[['match_id', 'start', 'league', 'bo', 't1', 't2', 't1c', 't2c', 'w1', 'w2']], on='match_id')
    t['blue_is_t1'] = np.where(t.blue_code_ls == t.t1c, 1, np.where(t.blue_code_ls == t.t2c, 0, np.nan))
    s1 = t.w1 > t.w2
    # truth, as P(blue won) in {0,1}
    sweep = (t[['w1', 'w2']].min(axis=1) == 0) & (t[['w1', 'w2']].max(axis=1) > 0)
    t['y_sweep'] = np.where(sweep, np.where(s1, t.blue_is_t1, 1 - t.blue_is_t1), np.nan)
    last_no = t.groupby('match_id').number.transform('max')
    full = (t.w1 + t.w2 == last_no)
    t['y_final'] = np.where((t.number == last_no) & full & (t.w1 != t.w2), np.where(s1, t.blue_is_t1, 1 - t.blue_is_t1), np.nan)

    L = pd.read_parquet(f'{D}/pm_links_raw.parquet')
    L = L[(L.score >= 85) & L.gnum.notna() & L.winner_pm.notna()]
    L['win_name'] = np.where(L.winner_pm == L.out_a, L.team_a, np.where(L.winner_pm == L.out_b, L.team_b, None))
    L = L.dropna(subset=['win_name']).drop_duplicates(['match_id', 'gnum'])
    t = t.merge(L[['match_id', 'gnum', 'win_name']].assign(gnum=lambda x: x.gnum.astype(int)).rename(columns={'gnum': 'number'}), on=['match_id', 'number'], how='left')
    win_is_t1 = np.where(t.win_name == t.t1, 1, np.where(t.win_name == t.t2, 0, np.nan))
    t['y_pm'] = np.where(np.isnan(win_is_t1), np.nan, np.where(win_is_t1 == 1, t.blue_is_t1, 1 - t.blue_is_t1))
    t['s_old'] = t.apply(old_rule, axis=1)
    t.to_parquet(f'{D}/game_table_raw.parquet')
    print(t.shape, t[['y_sweep', 'y_final', 'y_pm', 'blue_is_t1']].notna().sum().to_dict())


def label():
    """Winner for every game: ground truth where we have it, else the two-stage rule."""
    import winner
    t = pd.read_parquet(f'{D}/game_table_raw.parquet')
    # priority sweep > final > pm: pm labels carry ~0.3 % link errors (side flips) against the other two
    n = t.groupby('match_id').number.transform('size')
    t['y_sweep'] = t.y_sweep.where(n == t.w1 + t.w2)  # drop series with remakes / missing games
    t['y_truth'] = t.y_sweep.fillna(t.y_final).fillna(t.y_pm)
    tr = t.dropna(subset=['y_truth']).assign(y=lambda x: x.y_truth)
    t['p1'] = winner.p_blue(winner.fit(tr), t)
    t['y_rule'] = (t.p1 > .5).astype(float)
    t['y_src'] = np.where(t.y_sweep.notna(), 'sweep', np.where(t.y_final.notna(), 'final', np.where(t.y_pm.notna(), 'pm', 'rule')))
    t['blue_win'] = t.y_truth.fillna(t.y_rule)
    ok = t.dropna(subset=['y_truth'])
    print('rule vs truth (in-sample for stage 1):', ((ok.y_rule == ok.y_truth).mean()).round(4), len(ok))
    for c in ['y_pm', 'y_sweep', 'y_final']:
        o = t.dropna(subset=[c]); print(f'  vs {c}: {(o.y_rule == o[c]).mean():.4f} n={len(o)}')
    # series-count check: inferred wins per team must equal the series score
    t['t1_won'] = np.where(t.blue_is_t1 == 1, t.blue_win, 1 - t.blue_win)
    s = t.groupby('match_id').agg(n=('number', 'size'), mx=('number', 'max'), k=('t1_won', 'sum'), w1=('w1', 'first'), w2=('w2', 'first'))
    s = s[(s.n == s.w1 + s.w2) & (s.mx == s.n)]
    print('series whose game-winner count matches the score:', (s.k == s.w1).mean().round(4), len(s))
    t.to_parquet(f'{D}/game_table.parquet')
    print(t.y_src.value_counts().to_dict())


if __name__ == '__main__':
    main() if sys.argv[1:] == ['raw'] else label()
