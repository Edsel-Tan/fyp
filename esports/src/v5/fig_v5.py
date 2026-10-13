"""Figures for v5's backtests (results/viz/fig_v5_*.png).

Historical execution backtest (v5/backtest_v5.py; Polymarket-v1 prints, eval 2026-02..04):
  fig_v5_hist_latency.png   gross and fee-regime net pp/share vs latency L, clip and gate, v5 / v4 / v2 (95 % CI band on gross)
  fig_v5_hist_cum.png       cumulative PnL (per-share units) over the eval months at L = 0 / 15 / 30 s, clip and gate
  fig_v5_hist_sharpe.png    per-game and annualised daily Sharpe (net) vs L
  fig_v5_hist_null.png      v5's gate and clip at L=0 vs the five shuffled-outcome nulls, the placebo, v4 and v2
Recorded order books (v5/book_sim.py; captured minute bars, 2026-09-17 onward, L = 22 s):
  fig_v5_book_cum.png       cumulative net $ PnL per strategy (one panel per strategy; random control in grey)
  fig_v5_book_sharpe.png    per-game and annualised daily Sharpe (net $) per strategy and model
  fig_v5_book_pp.png        net pp per share with a match-clustered 95 % CI per strategy and model
  fig_v5_book_sens.png      net pp per share vs latency and quote timing
"""
import os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

D, R = '../data', '../results'
OUT = f'{R}/viz'
C = {'q_v5': '#2a78d6', 'q_v4': '#eb6834', 'q_lr v2': '#1baf7a', 'q_v2': '#1baf7a', 'random': '#8a8984', 'null': '#b5b4ae'}
NAME = {'q_v5': 'v5', 'q_v4': 'v4', 'q_lr v2': 'v2', 'q_v2': 'v2', 'random': 'random'}
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'font.size': 10,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2, 'legend.frameon': False})


def zero(ax):
    ax.axhline(0, color=INK2, lw=1, zorder=1)


def save(fig, name, note, dy=-0.04):
    fig.text(0.01, dy, note, fontsize=8, color=INK2, ha='left', va='top')
    fig.savefig(f'{OUT}/{name}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def hist():
    b = pd.read_csv(f'{R}/backtest_v5.csv')
    b = b[b.sig.isin(['q_v5', 'q_v4', 'q_lr v2'])]
    sigs = ['q_v5', 'q_v4', 'q_lr v2']
    # 1. pp/share vs latency
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, st in zip(axs, ('clip', 'gate')):
        for s in sigs:
            x = b[(b.sig == s) & (b.strat == st)].sort_values('L')
            if s == 'q_v5':
                ax.fill_between(x.L, x.ci_lo, x.ci_hi, color=C[s], alpha=0.12, lw=0)
            ax.plot(x.L, x.gross_pp, color=C[s], marker='o', ms=6, label=f'{NAME[s]} gross')
            ax.plot(x.L, x.fee_regime_net_pp, color=C[s], ls='--', lw=1.5, marker='o', ms=4, mfc=SURF, label=f'{NAME[s]} net (fee regime)')
        zero(ax)
        ax.set_title('Clip: every frame, 1 share toward q' if st == 'clip' else 'Gate: θ chosen on Nov–Jan selection months')
        ax.set_xlabel('decision latency L (s)')
        ax.set_xticks([0, 5, 15, 30, 60, 120])
    axs[0].set_ylabel('PnL per share (pp)')
    axs[1].legend(loc='upper right', fontsize=8, ncol=1)
    fig.suptitle('Historical backtest: PnL per share vs latency (eval Feb–Apr 2026)', x=0.01, y=1.03, ha='left', fontsize=12, fontweight='bold')
    save(fig, 'fig_v5_hist_latency.png', 'Solid: gross; shaded: v5\'s 95 % match-bootstrap CI (v4 / v2 CIs are as wide). Dashed: net of observed fees, fee regime (Mar–Apr) only.')

    # 2. cumulative PnL
    T = pd.read_parquet(f'{D}/v5/bt_trades.parquet')
    T = T[T.sig.isin(sigs)]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for j, L in enumerate((0, 15, 30)):
        for i, st in enumerate(('clip', 'gate')):
            ax = axs[i, j]
            for s in sigs:
                x = T[(T.L == L) & (T.sig == s) & (T.strat == st)].sort_values('t_start')
                net = x.gross - x.fee.fillna(0)
                th = x.theta.iloc[0] if st == 'gate' and len(x) else None
                ax.plot(x.t_start, net.cumsum(), color=C[s], label=f'{NAME[s]}' + (f' (θ {th:.2f})' if th is not None else ''))
            zero(ax)
            ax.set_title(f'{st}, L = {L} s')
            ax.legend(fontsize=8, loc='upper left')
            if j == 0:
                ax.set_ylabel('cumulative net PnL (shares)')
    for ax in axs[1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.suptitle('Historical backtest: cumulative net PnL, 1 share per trade', x=0.01, y=1.03, ha='left', fontsize=12, fontweight='bold')
    save(fig, 'fig_v5_hist_cum.png', 'Net of observed taker fees (non-zero from 2026-03). x: game start time.')

    # 3. Sharpe vs latency
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for j, st in enumerate(('clip', 'gate')):
        for i, (col, lab) in enumerate((('sr_game_net', 'per-game Sharpe (net)'), ('sr_ann_net', 'annualised daily Sharpe (net)'))):
            ax = axs[i, j]
            for s in sigs:
                x = b[(b.sig == s) & (b.strat == st)].sort_values('L')
                ax.plot(x.L, x[col], color=C[s], marker='o', ms=6, label=NAME[s])
            zero(ax)
            ax.set_ylabel(lab)
            ax.set_title(f'{st}')
            ax.set_xticks([0, 5, 15, 30, 60, 120])
            if i == 1:
                ax.set_xlabel('decision latency L (s)')
    axs[0, 1].legend(fontsize=9)
    fig.suptitle('Historical backtest: Sharpe ratio vs latency', x=0.01, y=1.03, ha='left', fontsize=12, fontweight='bold')
    save(fig, 'fig_v5_hist_sharpe.png', 'Per game: mean / sd of each game\'s net PnL. Annualised: net PnL per calendar day (zero days included) x sqrt(365).')

    # 4. nulls and placebo
    a = pd.read_csv(f'{R}/backtest_v5.csv')
    a0 = a[a.L == 0]
    log = open(f'{R}/backtest_v5.log').read()
    pm = re.search(r'placebo .*?mean ([-\d.]+) pp, sd ([\d.]+)', log)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, st in zip(axs, ('clip', 'gate')):
        rows = [(NAME[s], a0[(a0.sig == s) & (a0.strat == st)].iloc[0], C[s]) for s in sigs]
        rows += [(f'null {k}', a0[(a0.sig == f'q_v5 null{k}') & (a0.strat == st)].iloc[0], C['null']) for k in range(1, 6)]
        for i, (lab, r, col) in enumerate(rows):
            ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=2)
            ax.plot(r.gross_pp, i, 'o', color=col, ms=8, mec=SURF, mew=2)
        if st == 'gate' and pm:
            i = len(rows)
            m, sd = float(pm.group(1)), float(pm.group(2))
            ax.plot([m - 2 * sd, m + 2 * sd], [i, i], color=C['null'], lw=2)
            ax.plot(m, i, 'D', color=C['null'], ms=7, mec=SURF, mew=2)
            rows.append(('placebo', None, None))
        ax.set_yticks(range(len(rows)), [r[0] for r in rows])
        ax.axvline(0, color=INK2, lw=1)
        ax.set_xlabel('gross PnL per share (pp), L = 0, 95 % CI')
        ax.set_title(st)
    axs[0].invert_yaxis()
    fig.suptitle('v5 against shuffled-outcome nulls and the placebo', x=0.01, y=1.03, ha='left', fontsize=12, fontweight='bold')
    save(fig, 'fig_v5_hist_null.png', 'Nulls: v5 refit on permuted train outcomes, same theta selection. Placebo: another game\'s v5 at the same minute (mean ± 2 sd, 20 draws).')


def book():
    if not os.path.exists(f'{R}/book_sim.csv'):
        return
    S = pd.read_csv(f'{R}/book_sim.csv')
    T = pd.read_parquet(f'{D}/v5/book_trades.parquet')
    S, T = S[S.scope == 'tradable'], T[T.scope == 'tradable']  # games whose live feed was available (book_sim.py)
    base = S[(S.L == 22) & (S.quote == 'next')]
    T = T[(T.L == 22) & (T['quote'] == 'next')]
    strats = ['clip', 'gate30', 'gate20', 'gate_guard', 'conv', 'endgame']
    title = {'clip': 'clip (every frame)', 'gate30': 'gate θ = 0.30', 'gate20': 'gate θ = 0.20', 'gate_guard': 'gate 0.30 + team guard',
             'conv': 'convergence exit', 'endgame': 'endgame (q ≥ 0.99)'}
    sigs = ['q_v5', 'q_v4', 'q_v2']
    rnd = T[T.strat == 'random'].sort_values('T')
    t0 = pd.to_datetime(T['T'].min(), unit='s', utc=True)
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7), sharex=True)
    for ax, st in zip(axs.flat, strats):
        if st == 'clip':
            ax.plot(pd.concat([pd.Series([t0]), pd.to_datetime(rnd['T'], unit='s', utc=True)]), np.r_[0, rnd.net_usd.cumsum()],
                    color=C['random'], ls='--', lw=1.5, label='random')
        for s in sigs:
            x = T[(T.sig == s) & (T.strat == st)].sort_values('T')
            if len(x):
                ax.plot(pd.concat([pd.Series([t0]), pd.to_datetime(x['T'], unit='s', utc=True)]), np.r_[0, x.net_usd.cumsum()],
                        color=C[s], label=f'{NAME[s]} ({len(x)})')
        zero(ax)
        ax.set_title(title[st])
        ax.legend(fontsize=8, loc='lower left')
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    for ax in axs[:, 0]:
        ax.set_ylabel('cumulative net PnL (USD)')
    fig.suptitle('Recorded Polymarket books: cumulative net PnL, 50 USD per decision (L = 22 s, 142 live-tradable games)', x=0.01, y=1.03, ha='left', fontsize=12, fontweight='bold')
    save(fig, 'fig_v5_book_cum.png', 'Captured minute-bar books from 2026-09-17; fill at the book ending the decision minute, 1 c walk beyond the touch, fee 0.05 p(1-p). (n) = trades.')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    w = 0.26
    xs = np.arange(len(strats))
    for ax, col, lab in ((axs[0], 'sr_game', 'per-game Sharpe (net USD)'), (axs[1], 'sr_ann', 'annualised daily Sharpe (net USD)')):
        for k, s in enumerate(sigs):
            v = [base[(base.sig == s) & (base.strat == st)][col].squeeze() if len(base[(base.sig == s) & (base.strat == st)]) else np.nan for st in strats]
            ax.bar(xs + (k - 1) * w, v, w - 0.03, color=C[s], label=NAME[s])
        r = base[base.strat == 'random'][col].squeeze()
        ax.axhline(r, color=C['random'], ls='--', lw=1.5, label='random')
        zero(ax)
        ax.set_xticks(xs, [title[s] for s in strats], rotation=20, ha='right')
        ax.set_ylabel(lab)
        ax.grid(axis='x', visible=False)
    axs[1].legend(fontsize=9)
    fig.suptitle('Recorded Polymarket books: Sharpe ratio by strategy and model (L = 22 s, live-tradable games)', x=0.01, y=1.03, ha='left', fontsize=12, fontweight='bold')
    save(fig, 'fig_v5_book_sharpe.png', 'Per game: mean / sd of net USD per game. Annualised: net USD per calendar day of the window (zero days included) x sqrt(365).', dy=-0.12)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for k, s in enumerate(sigs):
        for i, st in enumerate(strats):
            r = base[(base.sig == s) & (base.strat == st)]
            if not len(r) or r.n.iloc[0] == 0:
                continue
            r = r.iloc[0]
            x = i + (k - 1) * 0.22
            ax.plot([x, x], [r.net_lo, r.net_hi], color=C[s], lw=2)
            ax.plot(x, r.net_pp, 'o', color=C[s], ms=8, mec=SURF, mew=2, label=NAME[s] if i == 0 else None)
    r = base[base.strat == 'random'].iloc[0]
    ax.axhspan(r.net_lo, r.net_hi, color=C['random'], alpha=0.12, lw=0)
    ax.axhline(r.net_pp, color=C['random'], ls='--', lw=1.5, label='random (CI shaded)')
    zero(ax)
    ax.set_xticks(range(len(strats)), [title[s] for s in strats], rotation=15, ha='right')
    ax.set_ylabel('net PnL per share (pp), 95 % CI')
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=9, ncol=4, loc='upper left')
    fig.suptitle('Recorded Polymarket books: net edge per share (L = 22 s, live-tradable games)', x=0.01, y=1.03, ha='left', fontsize=12, fontweight='bold')
    save(fig, 'fig_v5_book_pp.png', 'CI: match-clustered bootstrap. Net of 0.05 p(1-p) taker fee and walk.', dy=-0.1)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, st in zip(axs, ('clip', 'gate30', 'gate_guard')):
        for s in sigs:
            x = S[(S.sig == s) & (S.strat == st) & (S.quote == 'next')].sort_values('L')
            ax.plot(x.L, x.net_pp, color=C[s], marker='o', ms=6, label=NAME[s])
            p = S[(S.sig == s) & (S.strat == st) & (S.quote == 'prev')]
            if len(p):
                ax.plot(p.L, p.net_pp, marker='^', ms=8, ls='none', color=C[s], mfc=SURF, mew=2)
        zero(ax)
        ax.set_title(title[st])
        ax.set_xlabel('decision latency L (s)')
        ax.set_xticks([0, 22, 60])
    axs[0].set_ylabel('net PnL per share (pp)')
    axs[0].legend(fontsize=9)
    fig.suptitle('Recorded Polymarket books: sensitivity to latency and quote timing', x=0.01, y=1.03, ha='left', fontsize=12, fontweight='bold')
    save(fig, 'fig_v5_book_sens.png', 'Circles: book at the end of the decision minute (conservative). Hollow triangles: the last book before it (optimistic, L = 22 s).')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    hist()
    book()
