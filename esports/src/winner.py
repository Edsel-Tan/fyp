"""Infer the winning side of a finished game.

There is no per-game winner field in the lolesports API. `p_blue` is a logistic
on end-of-game features from the 60 s frames (structure pushes over the last
60/120/180 s, end-state diffs): 99.83 % grouped-CV accuracy on 6,554 games with
ground truth.

`tail_side` (the side that took the last structure, from the last ~4 min
refetched at full resolution) was tried as a second stage for unsure games and
rejected: it was worse than the logistic in every confidence band.
"""
import datetime as dt
import numpy as np, pandas as pd
import lolapi
from collect_frames import flat, iso

F = ['push60', 'push120', 'push180', 'e_towers', 'e_inhibs', 'e_dead', 'kills60', 'e_gold_k']


def fit(t):
    from sklearn.linear_model import LogisticRegression
    X = t[F[:-1]].astype(float).assign(e_gold_k=t.e_gold / 1000)
    return LogisticRegression(C=0.1, max_iter=2000).fit(X, t.y.astype(int))


def p_blue(model, t):
    return model.predict_proba(t[F[:-1]].astype(float).assign(e_gold_k=t.e_gold / 1000))[:, 1]


def tail(gid, t_end, back=240):
    rows = []
    for k in range(-back // 10, 2):
        w = lolapi.window(gid, iso(t_end + dt.timedelta(seconds=10 * k)))
        if w and w.get('frames'):
            rows += [flat(f) for f in w['frames']]
    return pd.DataFrame(rows).drop_duplicates('ts') if rows else None


def tail_side(x):
    """+1 blue / -1 red / 0 unknown: who took the last structure in the tail."""
    if x is None or len(x) < 2:
        return 0
    s = (x.b_towers + x.b_inhibs).diff().fillna(0).to_numpy()
    r = (x.r_towers + x.r_inhibs).diff().fillna(0).to_numpy()
    ev = [(i, 1) for i in np.flatnonzero(s > 0)] + [(i, -1) for i in np.flatnonzero(r > 0)]
    if not ev:
        return 0
    last = max(i for i, _ in ev)
    sides = {sd for i, sd in ev if i == last}
    return sides.pop() if len(sides) == 1 else 0
