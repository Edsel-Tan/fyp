"""Trade v5 (and v4, v2) against the recorded Polymarket order books: the self-captured minute bars (~/fyp-pmlive/data/pmlive,
pm/pmlive_collect.py: per token and minute the book at the minute's last update: best bid / ask, touch sizes, 5-level depth),
2026-09-17 onward, on the games scored by v5/live_period.py.

Linking: per-game winner markets ("... - Game N Winner") of the captured lol-* events, matched to games as lollive's
pmmarkets.link does (slug team codes equal the game's codes, else both outcome names fuzzy-match >= 85; date within a day;
game number agrees); the blue token is the outcome matching the blue team. Links whose final captured price contradicts
blue_win are dropped.
Decisions: every 60 s frame at T = feed ts + L (L = 22 s = the live feed's median age; 0 and 60 s as sensitivity).
Quote: the book at the end of minute floor(T / 60) (`next`: 0-60 s after T, the conservative choice; `prev`, the last book
before T's minute, as sensitivity). Buying blue lifts the blue token's ask; selling blue buys the red token at its ask.
Fill: $50 stake; touch size at the ask, the rest 1 c through (if the 5-level depth covers it; else partial down to $5).
Fee 0.05 p (1 - p) per share (every lollive fill paid this schedule). Held to resolution unless the strategy exits at the bid.
Strategies (thresholds fixed in advance: the live tracker's config, none tuned here):
  clip        every frame, buy toward q vs the mid
  gate30      buy when q - ask > 0.30 (lollive's pre-registered v2_gate); gate20 at 0.20 (v5's selection-month theta at L=0)
  gate_guard  gate30 + both teams with >= 20 recent games + at most 2 entries per game (lollive v2_gate_guard)
  conv        gate at 0.05, sold at the bid once bid >= q - 0.02 or after 300 s (lollive v2_conv)
  endgame     buy the side q gives >= 0.99 when its ask <= 0.985 (lollive v2_endgame)
  random      a random side every frame (control; model-free)
Sharpe: per game (net $ per game, mean / sd) and daily (net $ per calendar day of the window, zero days included, x sqrt(365)).
Writes data/v5/book_trades.parquet, results/book_sim.csv, results/book_sim.log (stdout).
"""
import os, sys, re, glob, sqlite3
import numpy as np, pandas as pd, pyarrow.dataset as ds
from rapidfuzz import fuzz
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

D = '../data'
PM = os.path.expanduser('~/fyp-pmlive/data')
LL = os.path.expanduser('~/fyp-pmlive/lollive')
STAKE, WALK, FEE = 50.0, 0.01, 0.05
MODELS = ['q_v5', 'q_v4', 'q_v2']
rng = np.random.default_rng(0)


def norm(s):
    s = (s or '').lower().replace('dn freecs', 'dn soopers').replace('esports', '').replace('gaming', '').replace('team ', '').replace('e-sports', '')
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()


def sim(a, b):
    a, b = norm(a), norm(b)
    return max(fuzz.ratio(a, b), fuzz.token_set_ratio(a, b) - 5, 100 if a and a == b else 0)


def team_names():
    """esportsTeamId (livestats) -> (code, name), from the study's game table and lollive's followed games."""
    t = pd.read_parquet(f'{D}/game_table.parquet', columns=['blue_id', 'red_id', 'blue_code_ls', 'red_code_ls', 't1', 't2', 't1c', 't2c', 't_start'])
    out = {}
    for _, r in t.sort_values('t_start').iterrows():
        nm = {r.t1c: r.t1, r.t2c: r.t2}
        out[r.blue_id] = (r.blue_code_ls, nm.get(r.blue_code_ls))
        out[r.red_id] = (r.red_code_ls, nm.get(r.red_code_ls))
    return out


def markets():
    c = sqlite3.connect(f'file:{PM}/pmlive.sqlite?mode=ro', uri=True)
    t = pd.read_sql("""select t.tok, t.outcome, m.question, m.condition_id, e.slug from tokens t join markets m using(condition_id)
                       join events e on e.event_id = t.event_id where e.slug like 'lol-%' and m.question like '%Game % Winner%'""", c)
    t['game'] = t.question.str.extract(r'Game (\d) Winner').astype(float)
    s = t.slug.str.extract(r'^lol-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d\d-\d\d)')
    t['c1'], t['c2'], t['date'] = s[0], s[1], pd.to_datetime(s[2]).dt.date
    return t


def link(G, M, names):
    lol = sqlite3.connect(f'file:{LL}/data/paper.sqlite?mode=ro', uri=True)
    lg = pd.read_sql('select game_id, blue_code, red_code, blue_name, red_name from games', lol).set_index('game_id')
    rows = []
    mk = {cid: x.sort_values('tok') for cid, x in M.groupby('condition_id')}
    for g in G.itertuples():
        if g.game_id in lg.index and pd.notna(lg.at[g.game_id, 'blue_code']):
            bc, rc, bn, rn = lg.loc[g.game_id, ['blue_code', 'red_code', 'blue_name', 'red_name']]
        else:
            (bc, bn), (rc, rn) = names.get(g.blue_id, (None, None)), names.get(g.red_id, (None, None))
        if bn is None and bc is None:
            continue
        day = g.t_start.date()
        best = None
        for cid, x in mk.items():
            r0 = x.iloc[0]
            if r0.game != g.number or abs((r0.date - day).days) > 1 or len(x) != 2:
                continue
            by_code = {r0.c1, r0.c2} == {str(bc).lower(), str(rc).lower()}
            o = x.outcome.tolist()
            sb, sr = [sim(a, bn) for a in o], [sim(a, rn) for a in o]
            bi = 0 if sb[0] + sr[1] >= sb[1] + sr[0] else 1
            score = min(sb[bi], sr[1 - bi])
            if not by_code and score < 85:
                continue
            if by_code and score < 60:
                bi = 0 if r0.c1 == str(bc).lower() else 1
                score = 60
            key = (by_code, score)
            if best is None or key > best[0]:
                best = (key, cid, x.tok.iloc[bi], x.tok.iloc[1 - bi], r0.slug)
        if best:
            rows.append(dict(game_id=g.game_id, condition_id=best[1], tok_b=int(best[2]), tok_r=int(best[3]), slug=best[4], link_score=best[0][1], by_code=best[0][0]))
    return pd.DataFrame(rows)


def load_bars(toks):
    d = ds.dataset(glob.glob(f'{PM}/pmlive/bars/dt=*/*.parquet'), format='parquet')
    b = d.to_table(filter=ds.field('tok').isin(list(toks)), columns=['tok', 'ts', 'bb', 'ba', 'bsz', 'asz', 'bd5', 'ad5']).to_pandas()
    b = b.sort_values(['tok', 'ts']).drop_duplicates(['tok', 'ts'], keep='last')
    for c in ('bb', 'ba'):
        b[c] = b[c] / 1000
    return {k: x.set_index('ts') for k, x in b.groupby('tok')}


def quote(bars, tok, minute):
    x = bars.get(tok)
    if x is None:
        return None
    try:
        r = x.loc[minute]
    except KeyError:
        return None
    return r


def fill_buy(r, stake=STAKE):
    """Shares and VWAP for buying `stake` USD at the book in bar r (touch, then 1 c through within the 5-level depth)."""
    if r is None or not (0 < r.ba < 1):
        return 0.0, np.nan
    sh = stake / r.ba
    if r.asz >= sh:
        return sh, r.ba
    if r.ad5 >= sh:
        return sh, (r.asz * r.ba + (sh - r.asz) * min(r.ba + WALK, 0.999)) / sh
    sh = r.ad5
    if sh * r.ba < 5:
        return 0.0, np.nan
    return sh, (r.asz * r.ba + (sh - r.asz) * min(r.ba + WALK, 0.999)) / sh


def fill_sell(r, sh):
    if r is None or not (0 < r.bb < 1):
        return np.nan
    if r.bsz >= sh:
        return r.bb
    return (r.bsz * r.bb + (sh - r.bsz) * max(r.bb - WALK, 0.001)) / sh


def books_at(F, bars, L, mode):
    """Per frame: blue and red token books at the decision minute."""
    T = ((F.ts - pd.Timestamp(0, tz='UTC')).dt.total_seconds().astype('int64') + L).to_numpy()
    mi = T // 60 - (1 if mode == 'prev' else 0)
    qb, qr = [], []
    for tb, tr_, m in zip(F.tok_b.to_numpy(), F.tok_r.to_numpy(), mi):
        qb.append(quote(bars, tb, m)); qr.append(quote(bars, tr_, m))
    F = F.assign(T=T, minute=mi)
    F['bid'] = [r.bb if r is not None and r.bb > 0 else np.nan for r in qb]
    F['ask'] = [r.ba if r is not None and 0 < r.ba < 1 else np.nan for r in qb]
    F['r_ask'] = [r.ba if r is not None and 0 < r.ba < 1 else np.nan for r in qr]
    F['r_ask'] = F.r_ask.fillna(1 - F.bid)
    F['qbk'], F['qrk'] = qb, qr
    F = F[F.bid.notna() & F.ask.notna() & (F.ask > F.bid)].copy()
    F['mid'] = (F.bid + F.ask) / 2
    return F


def run(F, bars, sig, strat, L):
    out = []
    open_n = {}
    pos = []  # open convergence positions: dict(game, tok, side, sh, vwap, fee, t0, q)
    for r in F.itertuples():
        q = getattr(r, sig) if sig != 'random' else np.nan
        side = 0
        if strat == 'clip':
            side = 1 if q > r.mid else -1
        elif strat in ('gate30', 'gate20', 'gate_guard'):
            th = 0.20 if strat == 'gate20' else 0.30
            side = 1 if q - r.ask > th else (-1 if (1 - q) - r.r_ask > th else 0)
            if side and strat == 'gate_guard':
                if r.n_min_games < 20 or open_n.get(r.game_id, 0) >= 2:
                    side = 0
        elif strat == 'conv':
            side = 1 if q - r.ask > 0.05 else (-1 if (1 - q) - r.r_ask > 0.05 else 0)
        elif strat == 'endgame':
            side = 1 if (q >= 0.99 and r.ask <= 0.985) else (-1 if (q <= 0.01 and r.r_ask <= 0.985) else 0)
        elif strat == 'random':
            side = 1 if rng.random() < 0.5 else -1
        if strat == 'conv':  # exits first (held positions in this game)
            for p in [p for p in pos if p['game'] == r.game_id]:
                qb = r.qbk if p['side'] == 1 else r.qrk
                qx = q if p['side'] == 1 else 1 - q
                if qb is not None and qb.bb > 0 and (qb.bb >= qx - 0.02 or r.T - p['t0'] >= 300):
                    px = fill_sell(qb, p['sh'])
                    if not np.isnan(px):
                        p.update(exit_px=px, exit_fee=FEE * px * (1 - px), exit_T=r.T, exit_reason='conv' if qb.bb >= qx - 0.02 else 'timeout')
                        out.append(p); pos.remove(p)
        if side == 0:
            continue
        if strat == 'conv' and any(p['game'] == r.game_id for p in pos):
            continue  # one convergence position per game at a time
        sh, px = fill_buy(r.qbk if side == 1 else r.qrk)
        if sh <= 0:
            continue
        open_n[r.game_id] = open_n.get(r.game_id, 0) + 1
        t = dict(game=r.game_id, match_id=r.match_id, league=r.league, t_start=r.t_start, T=r.T, clock=r.clock, side=side, sh=sh, vwap=px,
                 fee=FEE * px * (1 - px), q=q, mid=r.mid, y=r.blue_win, t0=r.T)
        if strat == 'conv':
            pos.append(t)
        else:
            out.append(t)
    out += pos  # never exited: held to resolution
    if not out:
        return pd.DataFrame()
    o = pd.DataFrame(out)
    if 'exit_px' not in o:
        o['exit_px'], o['exit_fee'], o['exit_reason'] = np.nan, np.nan, None
    win = np.where(o.side == 1, o.y, 1 - o.y)
    payoff = np.where(o.exit_px.notna(), o.exit_px, win)
    o['exit_reason'] = o.exit_reason.fillna('resolution')
    o['gross_sh'] = payoff - o.vwap
    o['net_sh'] = o.gross_sh - o.fee - o.exit_fee.fillna(0)
    o['gross_usd'], o['net_usd'] = o.gross_sh * o.sh, o.net_sh * o.sh
    o['day'] = pd.to_datetime(o['T'], unit='s', utc=True).dt.floor('D')
    return o.assign(sig=sig, strat=strat, L=L)


def summarize(o, days):
    if not len(o):
        return dict(n=0)
    g = o.groupby('match_id').agg(s=('net_sh', 'sum'), sg=('gross_sh', 'sum'), n=('net_sh', 'size'))
    k = rng.integers(0, len(g), (2000, len(g)))
    bs = g.s.to_numpy()[k].sum(1) / g.n.to_numpy()[k].sum(1)
    pg = o.groupby('game').net_usd.sum()
    dd = o.groupby('day').net_usd.sum().reindex(days, fill_value=0.0)
    cum = o.sort_values('T').net_usd.cumsum()
    return dict(n=len(o), games=o.game.nunique(), gross_pp=100 * o.gross_sh.mean(), net_pp=100 * o.net_sh.mean(),
                net_lo=100 * np.percentile(bs, 2.5), net_hi=100 * np.percentile(bs, 97.5),
                gross_usd=o.gross_usd.sum(), net_usd=o.net_usd.sum(), staked_usd=(o.sh * o.vwap).sum(),
                sr_game=pg.mean() / pg.std() if len(pg) > 2 else np.nan,
                sr_ann=dd.mean() / dd.std() * np.sqrt(365) if dd.std() > 0 else np.nan,
                max_dd=float((cum - cum.cummax()).min()), hit=(o.net_sh > 0).mean())


def check_against_lollive(bars, Lk):
    """Captured bars vs the live tracker's own book snapshots at its ticks (blue ask, same token)."""
    c = sqlite3.connect(f'file:{LL}/data/paper.sqlite?mode=ro', uri=True)
    t = pd.read_sql('select game_id, t, blue_ask, blue_bid from ticks where blue_ask is not null', c).merge(Lk[['game_id', 'tok_b']], on='game_id')
    res = {}
    for mode, off in (('same minute (end)', 0), ('previous minute (end)', -1)):
        d = []
        for r in t.itertuples():
            q = quote(bars, r.tok_b, int(r.t // 60) + off)
            if q is not None and 0 < q.ba < 1:
                d.append(abs(q.ba - r.blue_ask))
        d = np.array(d)
        res[mode] = (len(d), np.median(d) * 100 if len(d) else np.nan, (d <= 0.01).mean() if len(d) else np.nan)
    # (an offset scan on the mid puts the minimum at -1 / 0 minutes, median |diff| 1.5 c: the clocks agree; the rest is
    # movement within the minute)
    return res


def main():
    pd.set_option('display.width', 250)
    F = pd.read_parquet(f'{D}/v5/live_period_preds.parquet')
    G = pd.read_parquet(f'{D}/v5/live_period_games.parquet')
    assert G.number.notna().all()
    M = markets()
    Lk = link(G[G.game_id.isin(F.game_id)], M, team_names())
    bars = load_bars(set(Lk.tok_b) | set(Lk.tok_r))
    # resolution check: median captured blue-token mid from the game's last frame to 2 h after it (a market's last bars can be a
    # frozen post-close book, so the last captured price is not the resolution)
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
    print(f'games scored {F.game_id.nunique()}; captured per-game markets {M.condition_id.nunique()}; linked {len(Lk)} '
          f'(by code {Lk.by_code.sum()}); with bars {Lk.tok_b.isin(bars).sum()}; resolved in capture {res.sum()}, contradicting blue_win {bad.sum()} (dropped)')
    Lk = Lk[~bad & Lk.tok_b.isin(bars)]
    print('linked games by league:', Lk.league.value_counts().to_dict())
    for k, v in check_against_lollive(bars, Lk).items():
        print(f'bars vs lollive tick snapshots, blue ask, {k}: n {v[0]}, median |diff| {v[1]:.2f} c, within 1 c {v[2]:.1%}')
    F = F.merge(Lk[['game_id', 'tok_b', 'tok_r']], on='game_id').sort_values(['ts']).reset_index(drop=True)
    # live-tradable: the live feed was available while the game was played. From the live tracker's start (2026-09-30 16:32 UTC),
    # games it followed to the end on a working feed (status `finished`); before that, every league except WSCI (its streams
    # have statsStatus disabled). WSCI and the stats-disabled Demacia Cup / CBLOL / VCS games only have a feed after the series.
    lol = sqlite3.connect(f'file:{LL}/data/paper.sqlite?mode=ro', uri=True)
    lg = pd.read_sql('select game_id, status, joined_at from games', lol)
    t_live = pd.to_datetime(lg.joined_at.min(), unit='s', utc=True)
    fin = set(lg.game_id[lg.status == 'finished'])
    gs = G.set_index('game_id')
    trad = {g for g in F.game_id.unique() if (gs.at[g, 't_start'] < t_live and gs.at[g, 'league'] != 'wsci') or g in fin}
    print(f'live-tradable linked games: {len(trad)} of {F.game_id.nunique()} (tracker start {t_live:%Y-%m-%d %H:%M} UTC)')
    days = pd.date_range(F.ts.min().floor('D'), F.ts.max().floor('D'), freq='D')
    rows, allt = [], []
    for L, mode in ((22, 'next'), (0, 'next'), (60, 'next'), (22, 'prev')):
        Q0 = books_at(F, bars, L, mode)
        for scope in (('tradable', 'all') if (L, mode) == (22, 'next') else ('tradable',)):
            Q = Q0[Q0.game_id.isin(trad)] if scope == 'tradable' else Q0
            print(f'L={L}s quote={mode} scope={scope}: {len(Q)} frames with a two-sided book in {Q.game_id.nunique()} games; '
                  f'median spread {100 * (Q.ask - Q.bid).median():.1f} c', flush=True)
            for sig in MODELS + ['random']:
                for strat in (['clip', 'gate30', 'gate20', 'gate_guard', 'conv', 'endgame'] if sig != 'random' else ['random']):
                    o = run(Q, bars, sig, strat, L)
                    rows.append(dict(scope=scope, L=L, quote=mode, sig=sig, strat=strat, **summarize(o, days)))
                    if len(o):
                        allt.append(o.assign(quote=mode, scope=scope))
    R = pd.DataFrame(rows)
    print(R.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    T = pd.concat(allt, ignore_index=True)
    h = T[(T.scope == 'tradable') & (T.L == 22) & (T['quote'] == 'next') & (T.sig == 'q_v5')]
    print('\nv5, tradable games, L=22 s: net pp per share by league / concentration')
    for st in ('clip', 'gate30', 'gate20', 'gate_guard'):
        y = h[h.strat == st]
        g = y.groupby('game').net_usd.sum().sort_values()
        print(f'  {st}: top-3 games {g.tail(3).round(0).tolist()} = {g.tail(3).sum() / g.sum():.2f} of net $; worst-3 {g.head(3).round(0).tolist()}')
        print('   ', y.groupby('league').agg(n=('net_sh', 'size'), games=('game', 'nunique'), net_pp=('net_sh', lambda v: round(100 * v.mean(), 1)),
                                         net_usd=('net_usd', lambda v: round(v.sum()))).sort_values('n', ascending=False).to_dict('index'))
    R.to_csv(f'{D}/../results/book_sim.csv', index=False)
    T.drop(columns=['qbk', 'qrk'], errors='ignore').to_parquet(f'{D}/v5/book_trades.parquet')
    # forecast quality on the same frames (v5 vs v4 vs v2 vs the captured mid, L=22 s next-minute quote)
    from eval_pregame import ll
    Q = books_at(F, bars, 22, 'next')
    Q = Q[Q.game_id.isin(trad)]
    y = Q.blue_win.astype(int).to_numpy()
    print('\nper-game log loss on the tradable games\' frames with a book (L=22 s): ' + '  '.join(
        f'{c} {pd.Series(ll(y, Q[c].clip(1e-4, 1 - 1e-4).to_numpy()), index=Q.game_id).groupby(level=0).mean().mean():.4f}'
        for c in MODELS + ['mid']))


if __name__ == '__main__':
    main()
