"""v8 candidate, part 2: Baron / Elder buff state per 60 s frame (2026-10-16).

User question: after a team takes Baron or Elder, its win rate should be elevated while the buff lasts (Hand of Baron 180 s,
Aspect of the Elder 150 s) and the elevation should vanish when the buff ends. Who holds the buff matters, and a holder
who dies loses it.

v7's in-game model (v5's frozen logistic) only sees cumulative counts (`barons`, `elders`, blue minus red, x (1, m, m^2)),
so whatever a buff is worth is spread over the rest of the game. v4's group T had `bar_buf` / `eld_buf` (time left,
deaths ignored); greedy validation selection left it out of `sel window`.

Per frame (same rows as data/ingame.parquet, same clock as ingame_data.one; events dated at the midpoint between the
frame that first shows them and the one before, +-30 s at 60 s sampling):
  bar_t, eld_t     buff time left / duration, blue minus red (= v4's bar_buf / eld_buf)
  bar_n, eld_n     the same weighted by the share of the team still holding it: the holders are the players alive in the
                   frame that first shows the kill with no death since the previous frame; each holder drops out at their
                   next death (death counter increment). (holders alive) / 5 x time left / duration, blue minus red
  bar_on, eld_on   1 while any holder of that team still has the buff, blue minus red
  bar_post         1 during the 180 s after the team's Baron buff expired (the "it should disappear" window), blue minus red
  eld_post         the same for Elder (150 s after expiry)
  age_bar_b/r, age_eld_b/r   seconds since that side's latest Baron / Elder (NaN if none yet): for the event study
Also writes data/v8/buff_events.parquet: one row per Baron / Elder kill (game, clock, side, holders).
Usage: python v8/buffs.py
"""
import os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path[:0] = [SRC, os.path.join(SRC, 'v4')]
from features import raw_frames, drakes

D = os.path.join(SRC, '../data')
DUR = dict(bar=180.0, eld=150.0)


def buff_feats(d, clock):
    """d: filtered raw frames (features.raw_frames), clock: seconds. Returns (per-frame DataFrame, event list)."""
    n = len(d)
    c = np.asarray(clock, float)
    mid = np.r_[c[0], (c[1:] + c[:-1]) / 2]
    dth = np.stack([d[f'p{i}_deaths'].to_numpy(float) for i in range(1, 11)], 1)
    alive = np.stack([d[f'p{i}_currentHealth'].to_numpy(float) > 0 for i in range(1, 11)], 1)
    died = np.vstack([np.zeros((1, 10), bool), np.diff(dth, axis=0) > 0])
    cum = dict(bar=(d.b_barons.to_numpy(float), d.r_barons.to_numpy(float)),
               eld=(d.b_drakes.map(lambda s: drakes(s).count('elder')).to_numpy(float),
                    d.r_drakes.map(lambda s: drakes(s).count('elder')).to_numpy(float)))
    out, events = {}, []
    for typ, (cb, cr) in cum.items():
        T = DUR[typ]
        t_, n_, on, post = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        age = {0: np.full(n, np.nan), 1: np.full(n, np.nan)}
        for s, cs in ((0, cb), (1, cr)):
            sg = 1.0 if s == 0 else -1.0
            sl = slice(0, 5) if s == 0 else slice(5, 10)
            inc = np.r_[0, np.clip(np.diff(cs), 0, None)]
            for k in np.where(inc > 0)[0]:
                te = mid[k]
                hold = alive[k, sl] & ~died[k, sl]
                events.append(dict(type=typ, side='blue' if s == 0 else 'red', clock=te, frame_clock=c[k], holders=int(hold.sum())))
                h = hold.copy()
                for j in range(k, n):
                    a = c[j] - te
                    if j > k:
                        h &= ~died[j, sl]
                    if a < T:
                        t_[j] += sg * (T - a) / T
                        n_[j] += sg * h.sum() / 5 * (T - a) / T
                        if h.any():
                            on[j] = on[j] + sg if abs(on[j] + sg) <= 1 else on[j]
                    elif a < 2 * T:
                        post[j] = post[j] + sg if abs(post[j] + sg) <= 1 else post[j]
                    else:
                        break
                age[s][k:] = c[k:] - te
        out[f'{typ}_t'], out[f'{typ}_n'], out[f'{typ}_on'], out[f'{typ}_post'] = t_, n_, on, post
        out[f'age_{typ}_b'], out[f'age_{typ}_r'] = age[0], age[1]
    return pd.DataFrame(out), events


def one(gid, src='frames60'):
    from ingame_data import one as frames_one
    cwd = os.getcwd()
    try:
        os.chdir(SRC)  # ingame_data reads ../data relative to src
        o = frames_one(gid, src)
    finally:
        os.chdir(cwd)
    if o is None:
        return None
    d = raw_frames(pd.read_parquet(f'{D}/{src}/{gid}.parquet'))
    assert len(d) == len(o), gid
    f, ev = buff_feats(d, o.clock.to_numpy())
    return pd.concat([o[['game_id', 'ts']].reset_index(drop=True), f], axis=1), [dict(game_id=gid, **e) for e in ev]


def main():
    gids = pd.read_parquet(f'{D}/ingame.parquet', columns=['game_id']).game_id.unique()
    with ProcessPoolExecutor(40) as ex:
        res = [x for x in ex.map(one, gids, chunksize=64) if x is not None]
    f = pd.concat([r[0] for r in res], ignore_index=True)
    ev = pd.DataFrame([e for r in res for e in r[1]])
    os.makedirs(f'{D}/v8', exist_ok=True)
    f.to_parquet(f'{D}/v8/buffs.parquet')
    ev.to_parquet(f'{D}/v8/buff_events.parquet')
    print(f.shape, f.game_id.nunique(), 'events', ev.groupby(['type', 'side']).size().to_dict())
    print('holders at the kill:', ev.groupby('type').holders.describe().round(2).to_string())
    print('bar_t vs v4 bar_buf check follows in buff_eval')
    print(f.drop(columns=['game_id', 'ts']).describe().round(3).T.to_string())


if __name__ == '__main__':
    main()
