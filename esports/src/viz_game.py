"""Per-second model and market series for one game, for the replay video (render_replay.py).

The model is `prior_new oof` from eval_ingame2 (every stat and the out-of-fold prior, each x (1, m, m^2)), fit on the
train frames exactly as there, then scored on the game's full-resolution frames (data/frames10), resampled to 1 s.
Clock: with dense frames the 60 s anchor in ingame_data is wrong (the first gold frame is ~0:00, not 1:05), so the
clock is re-anchored at the first frame where total gold exceeds 5,000 (passive income starts at 1:05).
Market: the game's Polymarket fills, p_blue per trade, and a last-trade-price series.
Writes data/viz/{gid}.parquet (per second), data/viz/{gid}_fills.parquet, data/viz/{gid}_events.parquet, data/viz/{gid}.json.
Usage: python viz_game.py GAME_ID
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from eval_pregame import BURN, SPLIT
from eval_ingame import STATS, design
from ingame_data import one
from event_study import events

D = '../data'


def fit_model():
    f = pd.read_parquet(f'{D}/ingame.parquet')
    f = f[(f.t_start >= BURN) & (f.t_start < SPLIT) & (f.state != 'finished')].reset_index(drop=True)
    f = f.merge(pd.read_parquet(f'{D}/prior2.parquet'), on='game_id', how='left')
    X = design(f.assign(prior_logit=f.prior_new), STATS + ['prior'])
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X, f.blue_win.astype(int))


def main(gid):
    os.makedirs(f'{D}/viz', exist_ok=True)
    g = one(gid, 'frames10')
    t1 = g.ts[g.tot_gold > 5000].iloc[0]  # passive income starts at 1:05
    paused = (g.state == 'paused').to_numpy()
    dt = g.ts.diff().dt.total_seconds().fillna(0).to_numpy()
    run = np.cumsum(np.where(paused, 0, dt))
    g['clock'] = 65 + run - run[g.ts.searchsorted(t1)]
    prior = pd.read_parquet(f'{D}/prior2.parquet').set_index('game_id').prior_new[gid]
    g['prior_logit'] = prior
    m = fit_model()
    g['p_model'] = m.predict_proba(design(g, STATS + ['prior']))[:, 1]
    g['p_prior'] = 1 / (1 + np.exp(-prior))

    mk = pd.read_parquet(f'{D}/pm_game_markets.parquet')
    mk = mk[mk.game_id == gid].sort_values('vol').iloc[-1]
    fl = pd.read_parquet(f'{D}/pm_fills.parquet', filters=[('condition_id', '==', mk.condition_id)])
    fl['p_blue'] = fl.p_event if mk.seq1_blue == 1 else 1 - fl.p_event
    fl['t'] = pd.to_datetime(fl.ts, unit='s', utc=True).astype('datetime64[us, UTC]')
    fl = fl.sort_values('t')[['t', 'p_blue', 'usd', 'D', 'outcome_seq']]

    # per-second grid over the game, plus 10 min before (pre-game market)
    grid = pd.date_range(g.ts.iloc[0].floor('s') - pd.Timedelta(minutes=10), g.ts.iloc[-1].ceil('s'), freq='1s').astype('datetime64[us, UTC]')
    s = pd.DataFrame(index=grid)
    gg = g.assign(ts=g.ts.astype('datetime64[us, UTC]')).set_index('ts')[['clock', 'state', 'p_model', 'gold', 'kills', 'towers', 'drakes', 'barons', 'inhibs', 'alive']]
    s = pd.merge_asof(s, gg, left_index=True, right_index=True)
    ls = fl.set_index('t').p_blue.groupby(level=0).last()
    s['p_last'] = pd.merge_asof(s, ls.rename('p'), left_index=True, right_index=True).p
    # 30 s trailing VWAP: smoother than the last print, which bounces across the spread
    fu = fl.set_index('t')
    pv = (fu.p_blue * fu.usd).resample('1s').sum().reindex(grid, fill_value=0).rolling(30, min_periods=1).sum()
    vv = fu.usd.resample('1s').sum().reindex(grid, fill_value=0).rolling(30, min_periods=1).sum()
    s['p_vwap30'] = (pv / vv.replace(0, np.nan)).ffill().fillna(s.p_last)
    s['usd_1s'] = fu.usd.resample('1s').sum().reindex(grid, fill_value=0)
    s.index.name = 't'
    s.reset_index().to_parquet(f'{D}/viz/{gid}.parquet')
    fl.to_parquet(f'{D}/viz/{gid}_fills.parquet')
    ev = events(gid)
    ev['t'] = pd.to_datetime(ev.tau, unit='s', utc=True)
    ev['clock'] = np.interp(ev.tau, (g.ts - pd.Timestamp(0, tz='UTC')).dt.total_seconds(), g.clock)
    ev.to_parquet(f'{D}/viz/{gid}_events.parquet')
    meta = json.load(open(f'{D}/frames10/{gid}.meta.json'))
    gt = pd.read_parquet(f'{D}/game_table.parquet').set_index('game_id').loc[gid]
    tm = pd.read_parquet(f'{D}/games.parquet').set_index('game_id').loc[gid]
    code = {tm.blue_team_id: tm.blue_code, tm.red_team_id: tm.red_code}
    info = dict(game_id=gid, blue=code[meta['blueTeamMetadata']['esportsTeamId']], red=code[meta['redTeamMetadata']['esportsTeamId']],
                blue_win=float(gt.blue_win), league=str(gt.get('league', '')), market=mk.market_slug, vol=float(mk.vol),
                prior=float(1 / (1 + np.exp(-prior))), t_start=str(g.ts.iloc[0]), t_end=str(g.ts.iloc[-1]),
                champs={s: [(p['summonerName'], p['championId']) for p in meta[f'{s}TeamMetadata']['participantMetadata']]
                        for s in ('blue', 'red')})
    json.dump(info, open(f'{D}/viz/{gid}.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != 'champs'}))
    ig = s[s.clock.notna()]
    print(ig.iloc[::120][['clock', 'p_model', 'p_last', 'p_vwap30', 'gold', 'kills', 'towers']].to_string(float_format=lambda x: f'{x:.3f}'))
    print(ev.groupby('kind').size().to_dict())


if __name__ == '__main__':
    main(sys.argv[1])
