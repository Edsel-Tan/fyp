"""Is v2 overfit, or does live differ from the backtest for other reasons? (2026-10-05)

Live said the market beats v2 on log loss; the backtest said v2 beats the market. Tests:
  1. v2 overfitting: in-sample vs 5-fold (by match) CV within train vs the test split, and test log loss by quarter.
  2. The backtest market measure: last trade (what market_compare uses) vs a mid proxy built from the last taker-buy and
     taker-sell prints (pm_fills.D is ground-truth aggressor side), at the live lag (22 s).
  3. The backtest restricted to the live leagues, and by quarter (has the market caught up?).
  4. Live: per-game log loss of mid / v2 / v2f / v1 / gold-only (no prior) / prior-only, split by whether the game has a real
     prior (team_games_min >= 20) or not, and the chance of the live gap under the backtest's per-game distribution.
"""
import sqlite3, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eval_pregame import BURN, SPLIT, ll, boot_diff
from eval_ingame import STATS, design
from market_compare import market_at

D = '../data'
LIVE = os.path.expanduser('~/fyp-pmlive/lollive/data/paper.sqlite')
LAG = 22
rng = np.random.default_rng(0)


def sig(z):
    return 1 / (1 + np.exp(-z))


def pg(df, col, y='y'):
    return pd.Series(ll(df[y].to_numpy(), df[col].clip(1e-4, 1 - 1e-4).to_numpy()), index=df.game_id).groupby(level=0).mean()


def compare(df, cols, base, label, y='y'):
    g2m = df.drop_duplicates('game_id').set_index('game_id').match_id
    b = pg(df, base, y)
    print(f'  [{label}] games {df.game_id.nunique()} matches {df.match_id.nunique()} rows {len(df)}')
    for c in cols:
        a = pg(df, c, y)
        d, lo, hi = boot_diff(a.to_numpy(), b.reindex(a.index).to_numpy(), g2m.reindex(a.index).to_numpy())
        print(f'    {c:12s} {a.mean():.4f}   minus {base} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')


def part1():
    print('== 1. v2 overfitting: in-sample vs CV vs test ==')
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.state != 'finished')].reset_index(drop=True)
    f = f.merge(pd.read_parquet(f'{D}/prior2.parquet')[['game_id', 'prior_new']], on='game_id')
    y = f.blue_win.astype(int).to_numpy()
    X = design(f.assign(prior_logit=f.prior_new), STATS + ['prior'])
    tr = (f.t_start < SPLIT).to_numpy()
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    m = mk().fit(X[tr], y[tr])
    f['q_in'] = m.predict_proba(X)[:, 1]
    mids = f.match_id[tr].unique()
    fold = pd.Series(rng.integers(0, 5, len(mids)), index=mids)
    k = f.match_id.map(fold).to_numpy()
    f['q_cv'] = np.nan
    for j in range(5):
        a, b = tr & (k != j), tr & (k == j)
        f.loc[b, 'q_cv'] = mk().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    f['y'] = y
    ftr, fte = f[tr], f[~tr]
    print(f'  params {X.shape[1] + 1}, train frames {tr.sum()} games {ftr.game_id.nunique()}')
    print(f'  train in-sample per-game ll {pg(ftr, "q_in").mean():.4f}   train 5-fold CV {pg(ftr, "q_cv").mean():.4f}   '
          f'test (fit on train) {pg(fte, "q_in").mean():.4f}')
    # the prior itself: in-sample vs OOF on train (prior2 train rows are OOF already)
    fte = fte.assign(q=lambda d: d.q_in, qtr=fte.t_start.dt.tz_convert(None).dt.to_period('Q').astype(str))
    print('  test per-game ll by quarter:', {q: round(pg(d, 'q').mean(), 4) for q, d in fte.groupby('qtr')})


def mid_proxy(frames, fills, lag, win=60):
    """mid = (last taker-buy price + last taker-sell price) / 2, each within `win` s of ts + lag; NaN if either side is missing."""
    out = []
    for cid, fr in frames.groupby('condition_id'):
        fl = fills.get(cid)
        if fl is None:
            continue
        t = (fr.ts.dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[s]').astype('int64') + lag).to_numpy()
        fr = fr.copy()
        side = {}
        for s in (1, -1):
            g = fl[fl.Db == s]
            if g.empty:
                side[s] = np.full(len(t), np.nan)
                continue
            i = np.searchsorted(g.ts.to_numpy(), t, side='right') - 1
            p = np.where(i >= 0, g.p_blue.to_numpy()[np.maximum(i, 0)], np.nan)
            age = np.where(i >= 0, t - g.ts.to_numpy()[np.maximum(i, 0)], np.inf)
            side[s] = np.where(age <= win, p, np.nan)
        fr['mid'] = (side[1] + side[-1]) / 2
        fr['spr'] = side[1] - side[-1]
        out.append(fr)
    return pd.concat(out)


def part2(live_leagues):
    print(f'\n== 2-3. backtest at the live lag ({LAG} s): last trade vs mid proxy, by league and quarter ==')
    pr = pd.read_parquet(f'{D}/ingame_test_preds.parquet')
    mk = pd.read_parquet(f'{D}/pm_game_markets.parquet')
    fills = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event', 'D'])
    fills = fills[fills.condition_id.isin(mk.condition_id)].merge(mk[['condition_id', 'seq1_blue']], on='condition_id')
    fills['p_blue'] = np.where(fills.seq1_blue == 1, fills.p_event, 1 - fills.p_event)
    fills['Db'] = np.where(fills.seq1_blue == 1, fills.D, -fills.D)  # +1 = taker bought blue (paid the blue ask)
    fills = {c: g.sort_values('ts') for c, g in fills.groupby('condition_id')}
    mk = mk.sort_values('vol', ascending=False).drop_duplicates('game_id')
    fr = pr.merge(mk[['game_id', 'condition_id']], on='game_id')
    fr = market_at(fr, fills, LAG)
    fr = fr[(fr.mkt_age <= 60) & (fr.mkt > 0) & (fr.mkt < 1)]
    fr = mid_proxy(fr, fills, LAG)
    fr['y'] = fr.blue_win.astype(int)
    fr['q_v2'] = fr['q_lr v2']
    fr['q_nopr'] = fr['q_lr all']
    print(f'  median proxy spread {fr.spr.median():.3f}; frames with a mid proxy {fr.mid.notna().mean():.0%}')
    compare(fr, ['q_v2', 'q_gold'], 'mkt', 'all PM test games, market = last trade')
    both = fr[fr.mid.notna()]
    compare(both, ['mkt', 'q_v2', 'q_gold'], 'mid', 'same frames, market = mid proxy')
    sub = fr[fr.league.isin(live_leagues)]
    if len(sub):
        compare(sub, ['q_v2'], 'mkt', f'live leagues only {sorted(set(sub.league))}')
        print('    games by league:', sub.drop_duplicates('game_id').league.value_counts().to_dict())
    fr['qtr'] = fr.t_start.dt.tz_convert(None).dt.to_period('Q').astype(str)
    for q, d in fr.groupby('qtr'):
        if d.game_id.nunique() >= 30:
            compare(d, ['q_v2'], 'mkt', f'quarter {q}')
    return fr


def live():
    c = sqlite3.connect(f'file:{LIVE}?mode=ro', uri=True)
    g = pd.read_sql('select * from games', c)
    t = pd.read_sql('select * from ticks', c)
    g = g[g.blue_win_rule.notna()]
    t = t.merge(g[['game_id', 'match_id', 'league', 'blue_win_rule', 'team_games_min', 'prior_new']], on='game_id')
    t = t[t.blue_mid.notna() & (t.blue_mid > 0) & (t.blue_mid < 1)].copy()
    t['y'] = t.blue_win_rule.astype(int)
    t['q_prior'] = sig(t.prior_new)
    t['mid'] = t.blue_mid
    return g, t


def part4(fr):
    print('\n== 4. live: with vs without a real prior ==')
    g, t = live()
    t['low'] = t.team_games_min < 20
    cols = ['q_v2', 'q_v1', 'q_gold', 'q_prior']
    for lab, d in [('all live games', t), ('real prior (team_games_min >= 20)', t[~t.low]),
                   ('no real prior (team_games_min < 20)', t[t.low]),
                   ('real prior, EMEA Masters', t[~t.low & (t.league == 'emea_masters')]),
                   ('real prior, not EMEA Masters', t[~t.low & (t.league != 'emea_masters')]),
                   ('real prior, minute ticks only', t[~t.low & (t.trig == 'minute')])]:
        compare(d, cols, 'mid', lab)
    v = t[t.q_v2f.notna()]
    compare(v, ['q_v2', 'q_v2f'], 'mid', 'ticks with v2f (since 10-03)')
    compare(v[~v.low], ['q_v2', 'q_v2f'], 'mid', 'ticks with v2f, real prior')
    print(f'  low-prior games: {sorted(t[t.low].drop_duplicates("game_id").league.unique())}; q_v3 ticks: {t.q_v3.notna().sum()}')

    # 5. is the live gap consistent with the backtest? resample live-sized game sets from the backtest
    print('\n== 5. live gap vs the backtest distribution ==')
    lv = t[~t.low]
    dl = (pg(lv, 'q_v2') - pg(lv, 'mid')).mean()
    n = lv.game_id.nunique()
    bt = fr[fr.mid.notna()]
    for mcol in ('mkt', 'mid'):
        d = (pg(bt, 'q_v2') - pg(bt, mcol)).to_numpy()
        sims = np.array([rng.choice(d, n).mean() for _ in range(20000)])
        print(f'  backtest v2 - {mcol} at {LAG} s: mean {d.mean():+.4f} (games {len(d)}); live v2 - mid {dl:+.4f} (n {n}); '
              f'P(backtest {n}-game mean >= live) = {(sims >= dl).mean():.3f}')


if __name__ == '__main__':
    import sys
    if 'skip1' not in sys.argv:
        part1()
    _, t = live()
    fr = part2(set(t.league))
    part4(fr)
