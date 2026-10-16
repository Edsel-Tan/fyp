"""v7 on the self-captured books: a copy of v6/book_sim_v6.py run on v7/live_period_preds with q_v7 / q_v7m (archive weights from
v7/backtest_v7.py) added next to q_v6 / q_v6m. Original header:
v6 on the self-captured Polymarket books (v5/book_sim.py's linking, fills and strategies), plus v6m.

v6m = sigma(b_p logit(mid) + c_p logit(q_v6)) per game phase p (0-10 / 10-20 / 20-30 / 30+ min), no intercept, two weight sets:
  q_v6m      the archive weights (v6/backtest_v6.py: Polymarket-v1 selection months, thousands of games, print-based mid),
             at the archive latency nearest L. Chosen for the live tracker before this script scored anything: the captured
             fit set below has 63 games and gives implausible negative mid weights at 10-30 min.
  q_v6m_cap  weights fit on the captured games that started before the live tracker did (2026-09-17 .. 09-30 16:32 UTC);
             reported, not used live.
  The mid in the signal is the two-sided book at the end of the minute before the decision minute (`prev`: 0-60 s older than
  the decision instant T = feed ts + L), so the signal never sees the book it trades on. Live, the tracker's mid is the
  book at the decision instant, which is fresher. data/v6/comb.pkl: `live` (= archive) and `captured` weights.
Scored on the tradable games from the tracker's start (2026-09-30 16:32) to the end of v5/live_period's games (10-11):
  per-game log loss on the frames with a book (v6, v6m, v5, mid next / prev), and book_sim's strategies on
  v6, v6m, v5, v4, v2 at L = 22 s (next-minute quote; 0 / 60 s and prev quote as sensitivity), plus gateS: the gate with
  theta chosen per signal on the fit set (book_sim's gates use the tracker's fixed 0.30 / 0.20, which a market-shrunk
  signal like v6m rarely reaches).
Writes results/book_sim_v6.csv, data/v6/book_trades.parquet, data/v6/comb.pkl; stdout -> results/book_sim_v6.log.
"""
import os, sys, sqlite3, pickle
import numpy as np, pandas as pd, statsmodels.api as sm
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v5')]
import book_sim as B
from eval_pregame import ll

D = '../data'
PH = [0, 10, 20, 30, 1000]
SIGS = ['q_v7', 'q_v7m', 'q_v6', 'q_v6m', 'q_v5', 'q_v4', 'q_v2']
THETAS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30)
lg = lambda p: np.log(np.clip(p, .005, .995) / (1 - np.clip(p, .005, .995)))


def phase(d):
    return np.clip(np.searchsorted(PH, d.clock.to_numpy() / 60, side='right') - 1, 0, len(PH) - 2)


def comb_fit(x, sg='q_v6'):
    ph, out = phase(x), {}
    for p in range(len(PH) - 1):
        s = x[ph == p]
        r = sm.GLM(s.blue_win.astype(int).to_numpy(), np.c_[lg(s.mid_sig), lg(s[sg])], family=sm.families.Binomial()).fit()
        out[p] = np.r_[0.0, r.params]
    return out


def comb_apply(x, co, sg='q_v6'):
    ph, z = phase(x), np.zeros(len(x))
    for p, c in co.items():
        m = ph == p
        z[m] = c[0] + c[1] * lg(x.mid_sig.to_numpy()[m]) + c[2] * lg(x[sg].to_numpy()[m])
    return 1 / (1 + np.exp(-z))


def prev_mid(F, bars, L):
    """Two-sided blue mid at the end of the minute before T's minute (as-of within 5 min), per frame."""
    T = ((F.ts - pd.Timestamp(0, tz='UTC')).dt.total_seconds().astype('int64') + L).to_numpy()
    out = []
    for tb, m in zip(F.tok_b.to_numpy(), T // 60 - 1):
        x = bars.get(tb)
        v = np.nan
        if x is not None:
            i = x.index.searchsorted(m, side='right') - 1
            while i >= 0 and m - x.index[i] <= 5:
                r = x.iloc[i]
                if r.bb > 0 and 0 < r.ba < 1 and r.ba > r.bb:
                    v = (r.bb + r.ba) / 2
                    break
                i -= 1
        out.append(v)
    return np.array(out)


def choose(Q, bars, sig, L):
    best = (-1e9, None)
    for th in THETAS:
        o = run_th(Q, bars, sig, th, L)
        v = o.net_sh.mean() if len(o) >= 10 else -1e9
        best = max(best, (v, th), key=lambda z: z[0])
    return best[1]


def run_th(Q, bars, sig, th, L):
    """book_sim's gate at an arbitrary theta (B.run hard-codes 0.30 / 0.20)."""
    Q = Q.assign(sgl=Q[sig])
    side = np.where(Q.sgl - Q.ask > th, 1, np.where((1 - Q.sgl) - Q.r_ask > th, -1, 0))
    rows = []
    for r, sd in zip(Q.itertuples(), side):
        if sd == 0:
            continue
        sh, px = B.fill_buy(r.qbk if sd == 1 else r.qrk)
        if sh <= 0:
            continue
        rows.append(dict(game=r.game_id, match_id=r.match_id, league=r.league, T=r.T, clock=r.clock, side=sd, sh=sh, vwap=px,
                         fee=B.FEE * px * (1 - px), q=r.sgl, mid=r.mid, y=r.blue_win))
    if not rows:
        return pd.DataFrame(columns=['net_sh'])
    o = pd.DataFrame(rows)
    win = np.where(o.side == 1, o.y, 1 - o.y)
    o['gross_sh'] = win - o.vwap
    o['net_sh'] = o.gross_sh - o.fee
    o['gross_usd'], o['net_usd'] = o.gross_sh * o.sh, o.net_sh * o.sh
    o['day'] = pd.to_datetime(o['T'], unit='s', utc=True).dt.floor('D')
    o['exit_px'], o['exit_fee'], o['exit_reason'] = np.nan, np.nan, 'resolution'
    return o.assign(sig=sig, strat=f'gateS {th}', L=L)


def main():
    pd.set_option('display.width', 250)
    global ARCH
    ARCH = pickle.load(open(f'{D}/v6/comb_archive.pkl', 'rb'))
    ARCH7 = pickle.load(open(f'{D}/v7/comb_archive.pkl', 'rb'))
    assert ARCH['phases'] == PH
    F = pd.read_parquet(f'{D}/v7/live_period_preds.parquet')
    G = pd.read_parquet(f'{D}/v5/live_period_games.parquet')
    M = B.markets()
    Lk = B.link(G[G.game_id.isin(F.game_id)], M, B.team_names())
    bars = B.load_bars(set(Lk.tok_b) | set(Lk.tok_r))
    end = (F.groupby('game_id').ts.max() - pd.Timestamp(0, tz='UTC')).dt.total_seconds() // 60

    def endp(r):
        x = bars.get(r.tok_b)
        if x is None:
            return np.nan
        w = x.loc[int(end[r.game_id]):int(end[r.game_id]) + 120]
        return float(((w.bb + w.ba) / 2).where(w.ba > 0, w.bb).median()) if len(w) else np.nan
    Lk['last_blue'] = Lk.apply(endp, axis=1)
    Lk = Lk.merge(G[['game_id', 'blue_win', 'league']], on='game_id')
    res = Lk.last_blue.notna() & ((Lk.last_blue > 0.85) | (Lk.last_blue < 0.15))
    bad = res & ((Lk.last_blue > 0.5) != (Lk.blue_win == 1))
    Lk = Lk[~bad & Lk.tok_b.isin(bars)]
    F = F.merge(Lk[['game_id', 'tok_b', 'tok_r']], on='game_id').sort_values('ts').reset_index(drop=True)
    lol = sqlite3.connect(f'file:{B.LL}/data/paper.sqlite?mode=ro', uri=True)
    lgm = pd.read_sql('select game_id, status, joined_at from games', lol)
    t_live = pd.to_datetime(lgm.joined_at.min(), unit='s', utc=True)
    fin = set(lgm.game_id[lgm.status == 'finished'])
    gs = G.set_index('game_id')
    trad = {g for g in F.game_id.unique() if (gs.at[g, 't_start'] < t_live and gs.at[g, 'league'] != 'wsci') or g in fin}
    fit_g = {g for g in trad if gs.at[g, 't_start'] < t_live}
    ev_g = trad - fit_g
    print(f'linked {F.game_id.nunique()} games; tradable {len(trad)}: fit set (before the tracker, {t_live:%Y-%m-%d %H:%M}) {len(fit_g)}, '
          f'scored {len(ev_g)}')

    combs, Fs = {}, {}
    for L in (0, 22, 60, 120):
        X = F.assign(mid_sig=prev_mid(F, bars, L))
        fit = X[X.game_id.isin(fit_g) & X.mid_sig.notna()]
        co = comb_fit(fit)
        combs[L] = co
        X['q_v6m_cap'] = np.where(X.mid_sig.notna(), comb_apply(X.assign(mid_sig=X.mid_sig.fillna(.5)), co), np.nan)
        ca = ARCH['comb'][min(ARCH['comb'], key=lambda k: abs(k - L))]
        X['q_v6m'] = np.where(X.mid_sig.notna(), comb_apply(X.assign(mid_sig=X.mid_sig.fillna(.5)), ca), np.nan)
        ca7 = ARCH7['comb'][min(ARCH7['comb'], key=lambda k: abs(k - L))]
        X['q_v7m'] = np.where(X.mid_sig.notna(), comb_apply(X.assign(mid_sig=X.mid_sig.fillna(.5)), ca7, 'q_v7'), np.nan)
        Fs[L] = X
        print(f'L={L:3d}s v6m weights (mid, v6) by phase, {fit.game_id.nunique()} fit games / {len(fit)} frames: ' +
              '; '.join(f'{PH[p]}-{PH[p + 1] if PH[p + 1] < 1000 else "end"} {c[1]:+.2f} {c[2]:+.2f}' for p, c in co.items()))
    pickle.dump(dict(live=ARCH7, note='q_v7m = sigma(c0 + c1 logit(mid) + c2 logit(q_v7)) per phase (c0 = 0); comb keys = latency L (s); '
                          'the archive selection-month weights (v7/backtest_v7.py)'), open(f'{D}/v7/comb.pkl', 'wb'))

    # forecasts on the scored games' frames (L = 22 s): the signal mid is prev; mid_next is the book the trades use
    Q = B.books_at(Fs[22], bars, 22, 'next')
    Q = Q[Q.game_id.isin(ev_g) & Q.q_v6m.notna() & Q.q_v7m.notna()]
    y = Q.blue_win.astype(int).to_numpy()
    pgl = lambda c: pd.Series(ll(y, Q[c].clip(1e-4, 1 - 1e-4).to_numpy()), index=Q.game_id).groupby(level=0).mean()
    base = pgl('q_v5')
    g2m = Q.drop_duplicates('game_id').set_index('game_id').match_id
    from eval_pregame import boot_diff
    print(f'\nscored games, frames with both books at L=22 s: {len(Q)} frames, {Q.game_id.nunique()} games; per-game log loss (minus v5):')
    for c in ['q_v7', 'q_v7m', 'q_v6', 'q_v6m', 'q_v5', 'q_v4', 'q_v2', 'mid_sig', 'mid']:
        a = pgl(c)
        d = boot_diff(a.to_numpy(), base.reindex(a.index).to_numpy(), g2m.reindex(a.index).to_numpy())
        print(f'  {c:8s} {a.mean():.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]' + ('   (mid_sig: the prev-minute book v6m uses; mid: the decision book, '
              'up to 60 s after T)' if c == 'mid_sig' else ''))
    m = pgl('mid')
    for c in ('q_v7m', 'q_v6m', 'q_v7', 'q_v6', 'q_v5'):
        a = pgl(c)
        d = boot_diff(a.to_numpy(), m.reindex(a.index).to_numpy(), g2m.reindex(a.index).to_numpy())
        print(f'  {c} minus mid (decision book) {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]')
    Q['ph'] = phase(Q)
    for p, x in Q.groupby('ph'):
        yy = x.blue_win.astype(int).to_numpy()
        print(f'  phase {PH[p]}+ min ({len(x)} frames): ' + '  '.join(f'{c} {ll(yy, x[c].clip(1e-4, 1 - 1e-4).to_numpy()).mean():.4f}'
                                                              for c in ('q_v7m', 'q_v6m', 'q_v7', 'q_v6', 'q_v5', 'mid_sig', 'mid')))

    # trading
    days = pd.date_range(F.ts.min().floor('D'), F.ts.max().floor('D'), freq='D')
    rows, allt = [], []
    for L, mode in ((22, 'next'), (0, 'next'), (60, 'next'), (22, 'prev')):
        Q0 = B.books_at(Fs[min(Fs, key=lambda k: abs(k - L))], bars, L, mode)
        Qf = Q0[Q0.game_id.isin(fit_g)]
        Q = Q0[Q0.game_id.isin(ev_g)]
        Q = Q[Q.q_v6m.notna() & Q.q_v7m.notna()]
        for sig in SIGS + ['random']:
            strats = ['clip', 'gate30', 'gate20', 'gate_guard', 'conv', 'endgame'] if sig != 'random' else ['random']
            for strat in strats:
                o = B.run(Q, bars, sig, strat, L)
                rows.append(dict(L=L, quote=mode, sig=sig, strat=strat, **B.summarize(o, days)))
                if len(o):
                    allt.append(o.assign(quote=mode))
            if sig != 'random':
                th = choose(Qf[Qf[sig].notna()], bars, sig, L)
                o = run_th(Q, bars, sig, th, L)
                rows.append(dict(L=L, quote=mode, sig=sig, strat=f'gateS', theta=th, **B.summarize(o, days)))
                if len(o):
                    allt.append(o.assign(quote=mode))
        print(f'L={L} {mode}: {len(Q)} scored frames', flush=True)
    R = pd.DataFrame(rows)
    print(R.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    R.to_csv(f'{D}/../results/book_sim_v7.csv', index=False)
    T = pd.concat(allt, ignore_index=True).drop(columns=['qbk', 'qrk'], errors='ignore')
    T.to_parquet(f'{D}/v7/book_trades.parquet')
    h = T[(T.L == 22) & (T['quote'] == 'next')]
    print('\nL=22 next: v7 / v7m / v6 / v6m net $ concentration')
    for sig in ('q_v7', 'q_v7m', 'q_v6', 'q_v6m'):
        for st in sorted(h[h.sig == sig].strat.unique()):
            y_ = h[(h.sig == sig) & (h.strat == st)]
            g = y_.groupby('game').net_usd.sum().sort_values()
            if len(g) and g.sum() != 0:
                print(f'  {sig} {st}: {len(y_)} trades, {len(g)} games, net ${g.sum():.0f}; top-3 games {g.tail(3).round(0).tolist()}; worst-3 {g.head(3).round(0).tolist()}')


if __name__ == '__main__':
    main()
