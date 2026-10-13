"""Checks for v4/features.py (run from esports/src: python v4/test_features.py).

1. respawn(): known values (level 1 at 5:00 = 10 s; level 18 at 50:00 = 52.5 x 1.29 s; level 20 allowed).
2. Train/serve parity: the live tracker's saved feeds (lollive data/frames60) are dense (~0.2 s apart), training frames are 60 s apart.
   For each dense game, compute window_feats on the dense rows and on the rows thinned to one per 60 s, and compare them at the
   thinned timestamps. Level features (gold share, levels, HP, inhibitors down) must agree exactly; windowed ones (dg1, dk1, fight_g,
   5-min deltas) must agree closely (the dense lag frame is >= 50 s back, the thinned one 60 s); timing-based ones (respawn left,
   buffs) differ by at most the +-30 s event dating of 60 s sampling. Fails loudly if a feature's dense/thinned correlation < 0.9.
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import respawn, raw_frames, window_feats, GROUPS, WINDOW_ONLY

LL = os.path.expanduser('~/fyp-pmlive/lollive')


def clock_of(d):
    """live.features' clock on a filtered frame table (anchor: first frame above 5,000 total gold = 1:05)."""
    g0 = d.b_gold.iloc[0] + d.r_gold.iloc[0]
    gt0 = 65 + max(0, g0 - 5000) / 20.4 if g0 < 5400 else 90
    dts = d._t.diff().dt.total_seconds().fillna(0).to_numpy()
    return gt0 + np.cumsum(np.where(d.state.to_numpy() == 'paused', 0, dts))


def test_respawn():
    assert respawn(1, 300) == 10
    assert abs(respawn(18, 3000) - 52.5 * (1 + 0.2175 + 10 * 0.0145)) < 1e-9
    assert respawn(20, 600) == 57.5 and respawn(25, 600) == 57.5
    assert respawn(10, 1000) > respawn(10, 600)
    print('respawn: ok')


def test_parity(n_games=40):
    fs = sorted(glob.glob(f'{LL}/data/frames60/*.parquet'))[:n_games]
    cols = [c for k in WINDOW_ONLY for c in GROUPS[k]]
    A, B = [], []
    for fp in fs:
        d = raw_frames(pd.read_parquet(fp), min_gold=5000)
        if len(d) < 200 or any(f'p{i}_currentHealth' not in d for i in range(1, 11)):
            continue
        c = clock_of(d)
        dense = window_feats(d, c)
        keep, last = [], -1e9
        for k, x in enumerate(c):
            if x - last >= 60:
                keep.append(k)
                last = x
        keep = np.array(keep)
        thin = window_feats(d.iloc[keep].reset_index(drop=True), c[keep])
        A.append(dense.iloc[keep][cols].reset_index(drop=True))
        B.append(thin[cols])
    A, B = pd.concat(A, ignore_index=True), pd.concat(B, ignore_index=True)
    print(f'parity on {len(A)} thinned frames from {len(fs)} dense live feeds')
    bad = []
    for c in cols:
        a, b = A[c].to_numpy(), B[c].to_numpy()
        same = np.mean(np.isclose(a, b))
        r = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float('nan')
        print(f'  {c:12s} identical {same:6.1%}  corr {r:6.3f}  sd dense {a.std():7.3f} thinned {b.std():7.3f}')
        if not (same > 0.999 or r >= 0.9):
            bad.append(c)
    assert not bad, f'train/serve parity fails for {bad}'
    print('parity: ok')


if __name__ == '__main__':
    test_respawn()
    test_parity()
