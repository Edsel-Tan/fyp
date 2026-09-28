"""Execution backtest on Polymarket per-game markets, test-period games.

Decision at frame ts + L (L = latency, s). Known quotes at that instant are the last blue-buy
aggressor print (ask proxy) and the last blue-sell aggressor print (bid proxy), each at most
60 s old. Execution is the share-weighted VWAP of our side's aggressor prints *after* ts + L until
CLIP USD has traded (within 60 s), never the print that formed the quote, so dust prints at stale
prices cannot fill us (memory: pm-execution-accounting). Positions are held to
resolution, 1 share each; pnl per share = y - px for a buy and px - y for a sell.

Signals q (P(blue)):
  model    q_lr all              gold  q_gold (PLAN's simple model)
  comb     sigma(a logit(mid) + b logit(q_lr all) + c), fit on the selection months only
Strategies:
  A clip   every frame, trade 1 share toward q (buy if q > mid, else sell)
  B gate   trade only if q - ask > theta (buy) or bid - q > theta (sell); theta picked on the selection months
Walk-forward: selection = markets before SEL_END, evaluation = after. Taker fees exist only from 2026-03;
fee regime results use each fill's observed fee per share on our side's prints in the same market.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

D = '../data'
SFX = os.environ.get('PREDS_SFX', '')  # e.g. _null1 to backtest a shuffled-outcome null run
SEL_END = pd.Timestamp('2026-02-01', tz='UTC')
FEE_START = pd.Timestamp('2026-03-01', tz='UTC')
rng = np.random.default_rng(0)
CLIP = 50.0  # USD per fill
THETAS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30)


def placebo(q, sig, seed=0):
    """Replace each frame's signal with one from a different game at the same game minute (and same eval set)."""
    r = np.random.default_rng(seed)
    q = q.copy()
    q['_m'] = (q.clock // 60).astype(int)
    out = q[sig].to_numpy().copy()
    for _, ix in q.groupby('_m').indices.items():
        out[ix] = q[sig].to_numpy()[r.permutation(ix)]
    q[sig + '_placebo'] = out
    return q


def logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1 - p))


def load():
    pr = pd.read_parquet(f'{D}/ingame_test_preds{SFX}.parquet')
    mk = pd.read_parquet(f'{D}/pm_game_markets.parquet').sort_values('vol', ascending=False).drop_duplicates('game_id')
    f = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event', 'D', 'usd', 'price', 'fee_usdc'])
    f = f[f.condition_id.isin(mk.condition_id)].merge(mk[['condition_id', 'seq1_blue', 'game_id']], on='condition_id')
    f['p_blue'] = np.where(f.seq1_blue == 1, f.p_event, 1 - f.p_event)
    f['buy_blue'] = np.where(f.seq1_blue == 1, f.D == 1, f.D == -1)
    f['fee_sh'] = f.fee_usdc / (f.usd / f.price)  # fee per share traded
    fills = {g: x.sort_values('ts') for g, x in f.groupby('game_id')}
    fr = pr[pr.game_id.isin(fills)].copy()
    fr['t'] = fr.ts.dt.tz_convert('UTC').dt.tz_localize(None).astype('datetime64[s]').astype('int64')
    return fr, fills


def quotes(fr, fills, L, clip=CLIP):
    """Known quotes at t + L and a size-aware fill: the share-weighted VWAP of our side's prints after t + L
    until `clip` USD has traded, within 60 s (else no fill)."""
    out = []
    for g, x in fr.groupby('game_id'):
        fl = fills[g]
        t = x.t.to_numpy() + L
        res = {}
        for side, m in (('ask', fl.buy_blue.to_numpy()), ('bid', ~fl.buy_blue.to_numpy())):
            if not m.any():
                res.update({side: np.full(len(t), np.nan), side + '_px': np.full(len(t), np.nan),
                            side + '_fee': np.full(len(t), np.nan), side + '_usd': np.full(len(t), np.nan)})
                continue
            ts, p = fl.ts.to_numpy()[m], fl.p_blue.to_numpy()[m]
            usd, sh = fl.usd.to_numpy()[m], (fl.usd / fl.price).to_numpy()[m]
            fee = fl.fee_usdc.to_numpy()[m]
            cu, cs, cp, cf = (np.r_[0, np.cumsum(v)] for v in (usd, sh, sh * p, fee))
            i = np.searchsorted(ts, t, side='right') - 1  # last print at or before decision
            res[side] = np.where((i >= 0) & (t - ts[np.clip(i, 0, None)] <= 60), p[np.clip(i, 0, None)], np.nan)
            j = i + 1  # first print after decision
            k = np.searchsorted(cu, cu[j] + clip, side='left')  # cu[k] - cu[j] >= clip  -> prints j..k-1
            ok = (k <= len(ts)) & (j < len(ts))
            kk = np.clip(k, 1, len(ts))
            ok &= ts[kk - 1] - t <= 60
            shs = cs[kk] - cs[j]
            res[side + '_px'] = np.where(ok, (cp[kk] - cp[j]) / np.where(shs > 0, shs, 1), np.nan)
            res[side + '_fee'] = np.where(ok, (cf[kk] - cf[j]) / np.where(shs > 0, shs, 1), np.nan)
            res[side + '_usd'] = np.where(ok, cu[kk] - cu[j], np.nan)
        out.append(x.assign(**res))
    q = pd.concat(out)
    q = q.dropna(subset=['ask', 'bid'])
    q['mid'] = (q.ask + q.bid) / 2
    return q


def run(q, sig, strat, theta=0.0):
    s = q[sig].to_numpy()
    if strat == 'A':
        side = np.where(s > q.mid, 1, -1)
    else:
        side = np.where(s - q.ask > theta, 1, np.where(q.bid - s > theta, -1, 0))
    px = np.where(side == 1, q.ask_px, np.where(side == -1, q.bid_px, np.nan))
    fee = np.where(side == 1, q.ask_fee, np.where(side == -1, q.bid_fee, np.nan))
    y = q.blue_win.to_numpy()
    gross = np.where(side == 1, y - px, np.where(side == -1, px - y, np.nan))
    usd = np.where(side == 1, q.ask_usd, np.where(side == -1, q.bid_usd, np.nan))
    r = q[['game_id', 'match_id', 't_start', 'clock', 'mid']].assign(side=side, gross=gross, fee=fee, px=px, usd=usd)
    return r[(r.side != 0) & r.gross.notna()]


def summarize(r, name):
    if len(r) == 0:
        return dict(name=name, n=0)
    g = r.groupby('match_id').gross.agg(['sum', 'size'])
    k = rng.integers(0, len(g), (2000, len(g)))
    bs = g['sum'].to_numpy()[k].sum(1) / g['size'].to_numpy()[k].sum(1)
    fr = r[r.t_start >= FEE_START]
    return dict(name=name, n=len(r), games=r.game_id.nunique(), gross_pp=100 * r.gross.mean(),
                ci_lo=100 * np.percentile(bs, 2.5), ci_hi=100 * np.percentile(bs, 97.5),
                pre_fee_gross_pp=100 * r[r.t_start < FEE_START].gross.mean(),
                fee_regime_gross_pp=100 * fr.gross.mean(), fee_regime_net_pp=100 * (fr.gross - fr.fee).mean())


def main(lags):
    fr, fills = load()
    rows = []
    for L in lags:
        q = quotes(fr, fills, L)
        sel, ev = q[q.t_start < SEL_END], q[q.t_start >= SEL_END]
        X = lambda d: np.c_[logit(d.mid), logit(d['q_lr all'])]
        cm = LogisticRegression(C=1e6, max_iter=1000).fit(X(sel), sel.blue_win.astype(int))
        q['q_comb'] = cm.predict_proba(X(q))[:, 1]
        sel, ev = q[q.t_start < SEL_END], q[q.t_start >= SEL_END]
        for sig in ['q_lr all', 'q_gold', 'q_comb']:
            rows.append(dict(L=L, strat='A', sig=sig, **summarize(run(ev, sig, 'A'), 'eval')))
            best = max(((summarize(run(sel, sig, 'B', th), 'sel').get('gross_pp', -99), th)
                        for th in THETAS), key=lambda z: z[0] if not np.isnan(z[0]) else -99)
            rows.append(dict(L=L, strat=f'B th={best[1]}', sig=sig, sel_gross_pp=best[0], **summarize(run(ev, sig, 'B', best[1]), 'eval')))
        if L == 0:
            for sig in ['q_lr all']:
                curve = [dict(theta=th, **summarize(run(ev, sig, 'B', th), 'eval')) for th in THETAS]
                print('eval theta curve (sensitivity only; theta is chosen on selection months):')
                print(pd.DataFrame(curve)[['theta', 'n', 'games', 'gross_pp', 'ci_lo', 'ci_hi']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
                pl = [summarize(run(placebo(ev, sig, k), sig + '_placebo', 'B', th), 'placebo')['gross_pp'] for k in range(20) for th in (0.1, 0.15, 0.2)]
                print(f'placebo (other game same minute), theta in .1/.15/.2 x 20 draws: mean {np.mean(pl):.3f} pp, sd {np.std(pl):.3f}')
                r0 = run(ev, sig, 'B', 0.15)
                r0['pbin'] = pd.cut(r0.px, [0, .2, .4, .6, .8, 1])
                r0['fav'] = np.where((r0.side == 1) == (r0.mid > .5), 'buy favourite', 'buy underdog')
                r0['minute'] = pd.cut(r0.clock / 60, [0, 10, 20, 30, 60])
                for c in ['pbin', 'fav', 'minute']:
                    print(r0.groupby(c, observed=True).gross.agg(n='size', pp=lambda x: 100 * x.mean()).to_string(float_format=lambda x: f'{x:.2f}'))
                print('executed print size usd: median', r0.usd.median().round(1), 'p25', r0.usd.quantile(.25).round(1))
        print(f'L={L}s frames sel {len(sel)} eval {len(ev)} comb weights {cm.coef_[0].round(3)}', flush=True)
    r = pd.DataFrame(rows)
    print(r.drop(columns='name').to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    r.to_csv(f'{D}/../results/backtest{SFX}.csv', index=False)


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]] or [0, 5, 15, 30, 60])
