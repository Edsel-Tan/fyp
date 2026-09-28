"""Model vs Polymarket per-game markets, on in-game frames of the test period.

Market price at a frame is the last trade at or before (frame ts + lag), as P(blue).
Only frames whose last trade is at most 60 s old are kept. Reports:
  - per-game mean log loss of market vs each model on identical frames
  - encompassing regression y ~ logit(market) + logit(model), match-clustered bootstrap CI on the model's weight
The lag shifts the market *later*, which gives the market extra time to know the frame's state
(lag > 0 is conservative for the model).
"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from eval_pregame import ll, boot_diff

D = '../data'
rng = np.random.default_rng(0)


def market_at(frames, fills, lag):
    """Attach last trade p_blue at ts + lag (per game market)."""
    out = []
    for cid, fr in frames.groupby('condition_id'):
        fl = fills.get(cid)
        if fl is None:
            continue
        t = (fr.ts.dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[s]').astype('int64') + lag).to_numpy()
        i = np.searchsorted(fl.ts.to_numpy(), t, side='right') - 1
        ok = i >= 0
        fr = fr[ok].copy(); i = i[ok]
        fr['mkt'] = fl.p_blue.to_numpy()[i]
        fr['mkt_age'] = t[ok] - fl.ts.to_numpy()[i]
        out.append(fr)
    return pd.concat(out)


def logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def main(lag=0):
    pr = pd.read_parquet(f'{D}/ingame_test_preds.parquet')
    mk = pd.read_parquet(f'{D}/pm_game_markets.parquet')
    fills = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event'])
    fills = fills[fills.condition_id.isin(mk.condition_id)].merge(mk[['condition_id', 'seq1_blue']], on='condition_id')
    fills['p_blue'] = np.where(fills.seq1_blue == 1, fills.p_event, 1 - fills.p_event)
    fills = {c: g.sort_values('ts') for c, g in fills.groupby('condition_id')}
    # one market per game: the most-traded
    mk = mk.sort_values('vol', ascending=False).drop_duplicates('game_id')
    fr = pr.merge(mk[['game_id', 'condition_id']], on='game_id')
    fr = market_at(fr, fills, lag)
    fr = fr[(fr.mkt_age <= 60) & (fr.mkt > 0) & (fr.mkt < 1)]
    y = fr.blue_win.astype(int).to_numpy()
    qs = [c for c in fr.columns if c.startswith('q_')]
    print(f'lag {lag}s: frames {len(fr)} games {fr.game_id.nunique()} matches {fr.match_id.nunique()}')
    per = {}
    for c in ['mkt'] + qs:
        per[c] = pd.Series(ll(y, fr[c].to_numpy()), index=fr.game_id).groupby(level=0).mean()
    g2m = fr.drop_duplicates('game_id').set_index('game_id').match_id
    for c in per:
        d, lo, hi = boot_diff(per[c].to_numpy(), per['mkt'].reindex(per[c].index).to_numpy(), g2m.reindex(per[c].index).to_numpy())
        print(f'  {c:14s} per-game ll {per[c].mean():.4f}   minus market {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
    # encompassing: does the model carry information beyond the price?
    for c in ['q_lr all', 'q_gold', 'q_prior']:
        X = np.c_[logit(fr.mkt), logit(fr[c])]
        w = LogisticRegression(C=1e6, max_iter=1000).fit(X, y).coef_[0]
        mids = fr.match_id.unique()
        grp = {m: np.flatnonzero(fr.match_id.to_numpy() == m) for m in mids}
        bs = []
        for _ in range(300):
            ix = np.concatenate([grp[m] for m in rng.choice(mids, len(mids))])
            bs.append(LogisticRegression(C=1e6, max_iter=1000).fit(X[ix], y[ix]).coef_[0])
        bs = np.array(bs)
        print(f'  encompass {c:10s}: w_mkt {w[0]:.3f} [{np.percentile(bs[:, 0], 2.5):.3f},{np.percentile(bs[:, 0], 97.5):.3f}]'
              f'  w_model {w[1]:.3f} [{np.percentile(bs[:, 1], 2.5):.3f},{np.percentile(bs[:, 1], 97.5):.3f}]')
    fr.to_parquet(f'{D}/market_frames_lag{lag}.parquet')


if __name__ == '__main__':
    for lag in map(int, sys.argv[1:] or ['0']):
        main(lag)
