"""Map per-game Polymarket markets to games and sides.

Output data/pm_game_markets.parquet: condition_id, game_id, seq1_blue (1 if the
outcome_seq=1 token pays when blue wins), plus volume. p_blue = p_event if
seq1_blue else 1 - p_event.
"""
import numpy as np, pandas as pd

D = '../data'


def main():
    L = pd.read_parquet(f'{D}/pm_links_raw.parquet')
    L = L[(L.score >= 85) & L.gnum.notna()].copy()
    L['number'] = L.gnum.astype(int)
    f = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'outcome_seq', 'outcome_label'])
    lab = f[f.outcome_seq == 1].drop_duplicates('condition_id').set_index('condition_id').outcome_label
    L['seq1_label'] = L.condition_id.map(lab)
    L['seq1_team'] = np.where(L.seq1_label == L.out_a, L.team_a, np.where(L.seq1_label == L.out_b, L.team_b, None))
    t = pd.read_parquet(f'{D}/game_table.parquet')
    x = L.merge(t[['game_id', 'match_id', 'number', 't1', 't2', 'blue_is_t1', 'blue_win', 'y_src', 'y_pm']],
                on=['match_id', 'number'])
    s1_t1 = np.where(x.seq1_team == x.t1, 1, np.where(x.seq1_team == x.t2, 0, np.nan))
    x['seq1_blue'] = np.where(np.isnan(s1_t1), np.nan, (s1_t1 == x.blue_is_t1).astype(float))
    x = x.dropna(subset=['seq1_blue'])
    # sanity: the market's own resolution must say blue won exactly when blue_win says so. A disagreement means
    # the linker attached outcome labels to the wrong teams (e.g. lol-bro1-hle-2026-04-27: inverted prices all game)
    # or the label is wrong; either way the market is dropped.
    nm = lambda s: s.fillna('').str.strip().str.casefold()
    known = x.winner_pm.notna()
    win_blue = np.where(nm(x.winner_pm) == nm(x.seq1_label), x.seq1_blue, 1 - x.seq1_blue)
    bad = known & (win_blue != x.blue_win)
    print('markets', len(x), 'resolved', int(known.sum()), 'resolution contradicts blue_win:', int(bad.sum()),
          x.market_slug[bad].tolist())
    x = x[~bad]
    x[['condition_id', 'market_slug', 'game_id', 'match_id', 'number', 'seq1_blue', 'vol', 'fills', 'blue_win']] \
        .to_parquet(f'{D}/pm_game_markets.parquet')


if __name__ == '__main__':
    main()
