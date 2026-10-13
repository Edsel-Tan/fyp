"""Static figures of the LoL study for the report: results/viz/fig_*.png.

  fig_market_lag      model minus market per-game log loss by market lag (v1 `lr all`, v2 `prior_new oof`), clustered CIs
  fig_backtest_lag    gated backtest pp/share by execution latency L, with the shuffled-outcome null range
  fig_event_study     Polymarket price move and trade intensity around feed-timestamped events
  fig_ingame_minute   in-game log loss by game minute per model
  fig_calibration     reliability of the model (all test frames, and PM frames) and of the market
  fig_pregame         pre-game log loss by feature set
  fig_literature      published in-game models refit on our split, delta vs ours
  fig_game_traces     model vs market through six high-volume test games
Colors: reference palette, light steps; the model is always slot 1 (blue) and the market slot 2 (orange).
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eval_pregame import ll, boot_diff
from market_compare import market_at

D, OUT = '../data', '../results/viz'
S1, S2, S3, S4, S5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a897f', '#e4e3df'
MODEL, MARKET = S1, S2
V2 = 'q2_prior_new oof'
plt.rcParams.update({'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF, 'font.family': 'DejaVu Sans',
                     'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'axes.edgecolor': MUTED, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                     'axes.titlelocation': 'left', 'legend.frameon': False, 'lines.linewidth': 2})


def save(fig, name, note=None):
    if note:
        fig.text(0.01, -0.02, note, fontsize=7.5, color=MUTED, va='top')
    fig.savefig(f'{OUT}/{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def market_frames(lags):
    cache = f'{D}/viz/market_lags.parquet'
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    pr = pd.read_parquet(f'{D}/ingame2_test_preds.parquet')[['game_id', 'match_id', 'ts', 'clock', 'blue_win', 'q_lr all', V2]]
    mk = pd.read_parquet(f'{D}/pm_game_markets.parquet').sort_values('vol', ascending=False).drop_duplicates('game_id')
    fl = pd.read_parquet(f'{D}/pm_fills.parquet', columns=['condition_id', 'ts', 'p_event'])
    fl = fl[fl.condition_id.isin(mk.condition_id)].merge(mk[['condition_id', 'seq1_blue']], on='condition_id')
    fl['p_blue'] = np.where(fl.seq1_blue == 1, fl.p_event, 1 - fl.p_event)
    fl = {c: g.sort_values('ts') for c, g in fl.groupby('condition_id')}
    fr = pr.merge(mk[['game_id', 'condition_id']], on='game_id')
    rows = []
    for lag in lags:
        m = market_at(fr, fl, lag)
        m = m[(m.mkt_age <= 60) & (m.mkt > 0) & (m.mkt < 1)]
        y = m.blue_win.astype(int).to_numpy()
        per = {c: pd.Series(ll(y, m[c].to_numpy()), index=m.game_id).groupby(level=0).mean() for c in ['mkt', 'q_lr all', V2]}
        g2m = m.drop_duplicates('game_id').set_index('game_id').match_id.reindex(per['mkt'].index).to_numpy()
        for c in ['q_lr all', V2]:
            d, lo, hi = boot_diff(per[c].to_numpy(), per['mkt'].to_numpy(), g2m)
            rows.append(dict(lag=lag, model=c, d=d, lo=lo, hi=hi, games=len(per['mkt'])))
        if lag == 0:
            m[['game_id', 'clock', 'blue_win', 'mkt', 'q_lr all', V2]].to_parquet(f'{D}/viz/market_lag0_frames.parquet')
    r = pd.DataFrame(rows)
    r.to_parquet(cache)
    return r


def fig_market_lag():
    r = market_frames([0, 5, 15, 30, 60, 120, 300])
    print(r.to_string(float_format=lambda x: f'{x:+.4f}'))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for c, col, lab, dx in [('q_lr all', S3, 'v1: all stats × time', -0.03), (V2, MODEL, 'v2: + draft/strength prior', 0.03)]:
        q = r[r.model == c]
        x = np.log10(q.lag + 5) + dx
        ax.errorbar(x, q.d, yerr=[q.d - q.lo, q.hi - q.d], color=col, marker='o', ms=6, capsize=0, lw=2, elinewidth=1.2,
                    mec=SURF, mew=1.5, label=lab)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(np.log10(np.array([0, 5, 15, 30, 60, 120, 300]) + 5))
    ax.set_xticklabels(['0', '5', '15', '30', '60', '120', '300'])
    ax.set_xlabel('market lag: price taken this many seconds after the feed frame')
    ax.set_ylabel('model − market log loss (per game)')
    ax.text(0.99, 0.03, 'below 0: model more accurate', transform=ax.transAxes, ha='right', color=MUTED, fontsize=8.5)
    ax.set_title('The model leads Polymarket only until the market has had time to react')
    ax.legend(loc='upper left')
    save(fig, 'fig_market_lag', f'{int(r.games.iloc[0])} test-period games with a Polymarket per-game market; match-clustered bootstrap 95% CIs.')


def fig_backtest_lag():
    b = pd.read_csv('../results/backtest.csv')
    b = b[b.strat.str.startswith('B') & (b.sig == 'q_lr all')].sort_values('L')
    v2 = pd.read_csv('../results/backtest_v2.csv')
    v2 = v2[v2.strat.str.startswith('B') & (v2.sig == 'q_lr all')].sort_values('L')
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(b))
    ax.axhspan(-0.07, 0.32, color=GRID, alpha=.9, lw=0, zorder=0)
    ax.text(-0.35, -2.2, 'grey band: shuffled-outcome nulls (5 seeds)', ha='left', va='center', fontsize=8, color=MUTED)
    ax.errorbar(x - .08, b.gross_pp, yerr=[b.gross_pp - b.ci_lo, b.ci_hi - b.gross_pp], color=S3, marker='o', ms=6, lw=2,
                elinewidth=1.2, mec=SURF, mew=1.5, label='v1 model, gated θ=0.25–0.30')
    xv = x[:len(v2)] + .08
    ax.errorbar(xv, v2.gross_pp, yerr=[v2.gross_pp - v2.ci_lo, v2.ci_hi - v2.gross_pp], color=MODEL, marker='o', ms=6, lw=0,
                elinewidth=1.2, mec=SURF, mew=1.5, label='v2 model (L = 0, 5 s only)')
    for xi, (g, n) in enumerate(zip(b.gross_pp, b.n)):
        ax.text(xi - .14, g, f'{g:.1f}', ha='right', va='center', fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f'{l}' for l in b.L])
    ax.set_xlabel('execution latency L (s) after the feed frame')
    ax.set_ylabel('profit, pp per share (gross)')
    ax.set_title('Trading edge is significant only at zero latency and decays toward the null')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 0.93))
    save(fig, 'fig_backtest_lag', 'Eval months Feb–Apr 2026; fills at the share-VWAP of the next $50 of prints after decision + L; '
         'game-clustered 95% CIs. Primary cell t≈2.1, not Bonferroni-significant over 18 cells.')


def fig_event_study():
    e = pd.read_parquet(f'{D}/event_study.parquet')
    kinds = ['fight', 'tower', 'drake', 'baron', 'inhib']
    rng = np.random.default_rng(0)
    fig, axs = plt.subplots(2, 5, figsize=(12, 4.8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    for k, (a1, a2) in zip(kinds, axs.T):
        q = e[e.kind == k]
        n = q[['game_id', 'tau']].drop_duplicates().shape[0]
        for col, a, colr in [('move', a1, MARKET), ('intensity', a2, MUTED)]:
            p = q.pivot_table(index=['game_id', 'tau'], columns='bin', values=col)
            gids = p.index.get_level_values(0).to_numpy()
            ug = np.unique(gids); grp = {g: np.flatnonzero(gids == g) for g in ug}
            M = p.to_numpy()
            mean = np.nanmean(M, 0)
            bs = np.array([np.nanmean(M[np.concatenate([grp[g] for g in rng.choice(ug, len(ug))])], 0) for _ in range(200)])
            lo, hi = np.nanpercentile(bs, [2.5, 97.5], 0)
            a.fill_between(p.columns, lo, hi, color=colr, alpha=.2, lw=0)
            a.plot(p.columns, mean, color=colr if col == 'move' else INK2, lw=2 if col == 'move' else 1.5)
            a.axvline(0, color=INK2, lw=1, ls=':')
        a1.axhline(0, color=INK2, lw=.8)
        a1.set_title(f'{k} (n={n:,})', fontsize=10)
        a2.axhline(1, color=INK2, lw=.8)
    axs[0, 0].set_ylabel('Δ P(event side wins)')
    axs[1, 0].set_ylabel('trades vs pre-event')
    for a in axs[1]:
        a.set_xlabel('seconds from feed timestamp')
    for a in axs[0, 1:]:
        a.sharey(axs[0, 0])
    fig.suptitle('Polymarket reacts 5–10 s after the feed timestamp of a fight; objectives are priced in before they happen',
                 x=0.01, ha='left', fontweight='bold', fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save(fig, 'fig_event_study', 'Price move = VWAP of P(side) in 5 s bins minus the [−180, −120] s baseline, signed toward the event side. '
         'Game-clustered 95% bands. 1,534 games with full-resolution feed and market fills.')


def fig_ingame_minute():
    a = pd.read_csv('../results/ingame_eval.csv', index_col=0)
    b = pd.read_csv('../results/ingame2_eval.csv').set_index('model')
    cols = [c for c in a.columns if c.endswith('m')]
    x = np.array([int(c[:-1]) for c in cols]) + 2.5
    series = [('pre-game prior only', a.loc['prior', cols], MUTED), ('gold diff × time', a.loc['gold', cols], S4),
              ('gold + prior', a.loc['gold+prior', cols], S3), ('v1: all stats × time', a.loc['lr all', cols], S5),
              ('v2: + draft/strength prior', b.loc['prior_new oof', cols], MODEL)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for lab, y, c in series:
        ax.plot(x, y.to_numpy(float), color=c, marker='o', ms=4, mec=SURF, mew=1)
    ends = [(lab, float(y.iloc[-1]), c) for lab, y, c in series]
    ys = sorted(ends, key=lambda t: t[1])
    pos = []
    for lab, yv, c in ys:
        p = max(yv, pos[-1] + 0.022) if pos else yv
        pos.append(p)
        ax.text(x[-1] + 1, p, lab, va='center', fontsize=8.5, color=INK2)
    pg = {lab: v for lab, v in [('prior', a.loc['prior', 'per_game']), ('v2', b.loc['prior_new oof', 'per_game'])]}
    ax.set_xlim(0, x[-1] + 17)
    ax.set_xticks([0, 10, 20, 30, 40]); ax.set_xticklabels(['0', '10', '20', '30', '40+'])
    ax.set_xlabel('game minute (5-min buckets)')
    ax.set_ylabel('log loss (lower is better)')
    ax.set_title('In-game accuracy by minute: all stats with a time-varying prior weight is best throughout')
    save(fig, 'fig_ingame_minute', f'Test set 2025-07+, 10,582 games. Per-game mean log loss: prior {pg["prior"]:.3f}, '
         f'gold 0.480, v1 0.434, v2 {pg["v2"]:.3f}.')


def reliability(y, q, bins=10):
    b = np.clip((q * bins).astype(int), 0, bins - 1)
    d = pd.DataFrame(dict(y=y, q=q, b=b)).groupby('b').agg(q=('q', 'mean'), y=('y', 'mean'), n=('y', 'size'))
    return d


def fig_calibration():
    t = pd.read_parquet(f'{D}/ingame2_test_preds.parquet')
    m = pd.read_parquet(f'{D}/viz/market_lag0_frames.parquet')
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, title, parts in [
        (axs[0], 'All 10,582 test games', [('model v2', t.blue_win.to_numpy(int), t[V2].to_numpy(), MODEL)]),
        (axs[1], f'{m.game_id.nunique():,} games with a Polymarket market',
         [('model v2', m.blue_win.to_numpy(int), m[V2].to_numpy(), MODEL), ('Polymarket', m.blue_win.to_numpy(int), m.mkt.to_numpy(), MARKET)])]:
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls='--')
        for lab, y, q, c in parts:
            r = reliability(y, q)
            ax.plot(r.q, r.y, color=c, marker='o', ms=6, mec=SURF, mew=1.5, label=lab)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('predicted P(blue wins)')
        ax.set_aspect('equal'); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if len(parts) > 1:
            ax.legend(loc='upper left')
    axs[0].set_ylabel('observed blue win rate')
    fig.suptitle('Calibration: both model and market sit close to the diagonal', x=0.01, ha='left', fontweight='bold', fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, 'fig_calibration', 'Frame-level reliability, 10 equal-width bins. Market = last trade at the frame (≤ 60 s old). '
         'Model slope drifts from 1.0 to ~0.87 over the test year (recal.log).')


def fig_pregame():
    a = pd.read_csv('../results/pregame_eval.csv').set_index('model')
    b = pd.read_csv('../results/pregame2_eval.csv').set_index('model')
    rows = [('coin flip', a.loc['0.5', 'logloss'] if '0.5' in a.index else 0.6931), ('head-to-head', a.loc['h2h', 'logloss']),
            ('recent form (20)', a.loc['form_20', 'logloss']), ('one-hot player identity', a.loc['one-hot players (C=0.1)', 'logloss']),
            ('team Elo', a.loc['elo_team_64', 'logloss']), ('player Elo', a.loc['elo_play_32', 'logloss']),
            ('all features', a.loc['all features', 'logloss']), ('+ draft, lane & league strength', b.loc['all new', 'test_ll'])]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(rows))[::-1]
    v = np.array([r[1] for r in rows])
    ax.barh(y, v - 0.58, left=0.58, height=0.6, color=[MUTED] * 6 + [S3, MODEL])
    for yi, vi in zip(y, v):
        ax.text(vi + 0.001, yi, f'{vi:.3f}', va='center', fontsize=8.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.58, 0.70)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('test log loss (lower is better; axis starts at 0.58)')
    ax.set_title('Pre-game: ratings beat identities; draft and strength features add −0.0095')
    save(fig, 'fig_pregame', 'Test set 2025-07+, 10,584 games. AUC: team Elo 0.694, all features 0.720, all new 0.731.')


def fig_literature():
    l = pd.read_csv('../results/lit_compare.csv')
    l = l[l.model != 'ours lr all'].sort_values('d_vs_ours')
    names = {'silva lstm': 'Silva et al. 2018 LSTM', 'silva rnn': 'Silva et al. 2018 RNN', 'kim mlp DU-loss': 'Kim et al. 2020 MLP (DU loss)',
             'kim mlp': 'Kim et al. 2020 MLP', 'kim mlp + temp (T=0.98)': 'Kim et al. 2020 MLP + temp.',
             'riot/aws wp + prior': 'Riot/AWS WP features + our prior', 'snapshot@15 lr': 'snapshot LR @15 min',
             'ours lr no-prior': 'ours without prior', 'hodge per-minute lr': 'Hodge et al. per-minute LR',
             'snapshot@10 lr': 'snapshot LR @10 min', 'riot/aws wp (gbm)': 'Riot/AWS WP features (GBM)'}
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(l))[::-1]
    ax.errorbar(l.d_vs_ours, y, xerr=[l.d_vs_ours - l.lo, l.hi - l.d_vs_ours], fmt='o', color=MUTED, mec=SURF, ms=6, elinewidth=1.5)
    ax.axvline(0, color=MODEL, lw=2)
    ax.text(0.0005, -0.6, 'ours (v1)', color=INK2, fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([names.get(m, m) for m in l.model])
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Δ per-game log loss vs ours (right = worse)')
    ax.set_title('Published models refit on our data: architecture adds nothing; the prior does')
    save(fig, 'fig_literature', 'Same 60 s frames and split; match-clustered 95% CIs. Snapshot models are scored on their own frames only.')


def fig_game_traces():
    m = pd.read_parquet(f'{D}/viz/market_lag0_frames.parquet')
    mk = pd.read_parquet(f'{D}/pm_game_markets.parquet').sort_values('vol', ascending=False).drop_duplicates('game_id')
    g = pd.read_parquet(f'{D}/game_table.parquet').set_index('game_id')
    tm = pd.read_parquet(f'{D}/games.parquet').set_index('game_id')
    top = mk[mk.game_id.isin(m.game_id)].head(6)
    fig, axs = plt.subplots(2, 3, figsize=(12, 5.6), sharey=True)
    for ax, r in zip(axs.flat, top.itertuples()):
        q = m[m.game_id == r.game_id].sort_values('clock')
        t = tm.loc[r.game_id]
        code = {t.blue_team_id: t.blue_code, t.red_team_id: t.red_code}
        blue, red = code[g.loc[r.game_id, 'blue_id']], code[g.loc[r.game_id, 'red_id']]
        win = blue if g.loc[r.game_id, 'blue_win'] == 1 else red
        ax.axhline(.5, color=MUTED, lw=.8)
        ax.step(q.clock / 60, q.mkt, where='post', color=MARKET, label='Polymarket')
        ax.plot(q.clock / 60, q[V2], color=MODEL, label='model v2')
        date = '-'.join(r.market_slug.split('-')[-4:-1])
        ax.set_title(f'{blue} vs {red} · {date} · ${r.vol / 1e6:.1f}M · {win} won', fontsize=9.5)
        ax.set_ylim(0, 1)
    for ax in axs[:, 0]:
        ax.set_ylabel('P(blue wins)')
    for ax in axs[1]:
        ax.set_xlabel('game minute')
    axs[0, 0].legend(loc='lower left', fontsize=8.5)
    fig.suptitle('Model vs market through the six highest-volume test games (60 s frames)', x=0.01, ha='left', fontweight='bold', fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, 'fig_game_traces', 'Market = last trade at each feed frame (≤ 60 s old). Model v2 trained on 2024-04..2025-06 only.')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for f in [fig_market_lag, fig_backtest_lag, fig_event_study, fig_ingame_minute, fig_calibration, fig_pregame, fig_literature, fig_game_traces]:
        f()
