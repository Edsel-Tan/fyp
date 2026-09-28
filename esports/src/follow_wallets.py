"""Baseline: follow skilled takers (PLAN: "follow informed traders").

Selection: every linked LoL market (series and per-game) whose last trade is before SEL_END. Each taker's
per-share pnl is (seq1 won - p_event) * D, where D is the taker's ground-truth side on the p_event axis. Rank takers with at least
MIN_MKTS markets by a t-stat clustered by market; the top K are "skilled".
Evaluation: in per-game markets of eval-period games (the same as backtest.py), each taker fill by a skilled
wallet is copied, same direction, at the share-VWAP of the next $50 of same-side prints after its ts + L.
The unconditional version (copy every taker) is the control.
"""
import sys
import numpy as np, pandas as pd

D = '../data'
SEL_END = pd.Timestamp('2026-02-01', tz='UTC').timestamp()
MIN_MKTS, K, CLIP = 10, 50, 50.0
rng = np.random.default_rng(0)


def main(lags):
    mk = pd.read_parquet('../pm_lol_markets.parquet')[['condition_id', 'winning_outcome_label', 't1']]
    f = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event', 'D', 'usd', 'price', 'outcome_seq', 'outcome_label', 'taker'])
    lab1 = f[f.outcome_seq == 1].drop_duplicates('condition_id').set_index('condition_id').outcome_label
    mk['y1'] = (mk.condition_id.map(lab1) == mk.winning_outcome_label).astype(float)
    mk = mk[mk.condition_id.isin(lab1.index)]
    f = f.merge(mk[['condition_id', 'y1', 't1']], on='condition_id')
    f['pnl'] = (f.y1 - f.p_event) * f.D
    sel = f[f.t1 < SEL_END]
    per = sel.groupby(['taker', 'condition_id']).pnl.agg(['sum', 'size'])
    w = per.groupby(level=0).apply(lambda x: pd.Series(dict(m=len(x), pp=x['sum'].sum() / x['size'].sum(),
                                                          t=(x['sum'] / x['size']).mean() / ((x['sum'] / x['size']).std(ddof=1) / np.sqrt(len(x)) + 1e-9))))
    w = w[w.m >= MIN_MKTS].sort_values('t', ascending=False)
    top = set(w.index[:K])
    print(f'selection: {sel.condition_id.nunique()} markets, {len(w)} takers with >= {MIN_MKTS} markets; top {K} mean in-sample pp {100 * w.pp[:K].mean():.2f}')

    gm = pd.read_parquet(f'{D}/pm_game_markets.parquet').sort_values('vol', ascending=False).drop_duplicates('game_id')
    tp = pd.read_parquet(f'{D}/ingame_test_preds.parquet', columns=['game_id', 'match_id', 't_start']).drop_duplicates('game_id')
    tp = tp[tp.t_start >= pd.Timestamp(SEL_END, unit='s', tz='UTC')]
    gm = gm.merge(tp.drop(columns='match_id'), on='game_id')
    ev = f[f.condition_id.isin(gm.condition_id)].merge(gm[['condition_id', 'match_id']], on='condition_id').sort_values('ts')
    ev['sh'] = ev.usd / ev.price
    rows = []
    for L in lags:
        for who, sig in (('skilled', ev[ev.taker.isin(top)]), ('all takers', ev)):
            out = []
            for cid, s in sig.groupby('condition_id'):
                fl = ev[ev.condition_id == cid]
                for d in (1, -1):
                    sd = s[s.D == d]
                    if not len(sd):
                        continue
                    m = fl[fl.D == d]
                    ts, p, u, shr = m.ts.to_numpy(), m.p_event.to_numpy(), m.usd.to_numpy(), m.sh.to_numpy()
                    cu, cs, cp = np.r_[0, np.cumsum(u)], np.r_[0, np.cumsum(shr)], np.r_[0, np.cumsum(shr * p)]
                    t = sd.ts.to_numpy() + L
                    j = np.searchsorted(ts, t, side='right')
                    k = np.searchsorted(cu, cu[j] + CLIP, side='left')
                    ok = (j < len(ts)) & (k <= len(ts))
                    kk = np.clip(k, 1, len(ts))
                    ok &= ts[kk - 1] - t <= 60
                    px = (cp[kk] - cp[j]) / np.where(cs[kk] - cs[j] > 0, cs[kk] - cs[j], 1)
                    g = (sd.y1.to_numpy() - px) * d
                    out.append(pd.DataFrame(dict(match_id=sd.match_id.to_numpy()[ok], gross=g[ok])))
            r = pd.concat(out)
            gg = r.groupby('match_id').gross.agg(['sum', 'size'])
            kb = rng.integers(0, len(gg), (2000, len(gg)))
            bs = gg['sum'].to_numpy()[kb].sum(1) / gg['size'].to_numpy()[kb].sum(1)
            rows.append(dict(L=L, follow=who, n=len(r), matches=len(gg), gross_pp=100 * r.gross.mean(),
                             ci_lo=100 * np.percentile(bs, 2.5), ci_hi=100 * np.percentile(bs, 97.5)))
    r = pd.DataFrame(rows)
    print(r.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    r.to_csv(f'{D}/../results/follow_wallets.csv', index=False)


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]] or [0, 5, 15, 30, 60])
