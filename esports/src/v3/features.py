"""v3 step 7: encode a parsed minimap into a few blue-minus-red features (the "important features only" encoding).

Canonical map: x right, y down, blue fountain bottom-left (0, 1), red top-right (1, 0). The river is the x = y diagonal;
a point is on red's side when x > y. "Progress" of a champion towards the enemy base: u = (x - y + 1) / 2 for blue, 1 - u for red.
Pits (measured on the median background): baron (0.33, 0.30), dragon (0.68, 0.70).

Champions (detector peaks per team with p >= 0.6 (val2: per-team count exact 41 %, over 12 %, under 47 %; misses are piles), at most 5 per team; undetected champions contribute nothing):
  mm_vis       detected count, blue - red (also absorbs how much of the broadcast's minimap is readable)
  mm_adv       mean progress of detected blue - mean progress of detected red (0 if a side has none)
  mm_deep      champions on the enemy side of the river, blue - red
  mm_drk, mm_bar   champions within 0.12 of the dragon / baron pit, blue - red
  mm_group     largest number of a team's champions within 0.10 of each other, blue - red
Wards (team-coloured small blobs):
  mm_w         total, blue - red;  mm_wdeep  wards on the enemy side;  mm_wdrk, mm_wbar  wards within 0.15 of the pits
Objective x presence (with objectives.py): drake_up / elder_up x (mm_drk, mm_wdrk), baron_up x (mm_bar, mm_wbar)
Distances (added 2026-10-05 on the user's point that location, not count, decides a contest: "the team late to baron loses it"):
  d_bar, d_drk     red's minus blue's mean distance of its 3 nearest detected champions to the pit (> 0: blue is closer);
                   fewer than 3 detected -> mean of those found; none -> 0 and dd_miss = 1
  d1_bar, d1_drk   the same for the single nearest champion
  dbar_x_up, ddrk_x_up   d_bar x baron_up, d_drk x (drake_up + elder_up)
"""
import json
import numpy as np, pandas as pd

BARON, DRAGON = np.array([0.33, 0.30]), np.array([0.68, 0.70])
CH = ['mm_vis', 'mm_adv', 'mm_deep', 'mm_drk', 'mm_bar', 'mm_group']
WD = ['mm_w', 'mm_wdeep', 'mm_wdrk', 'mm_wbar']
INTER = ['drk_x_up', 'wdrk_x_up', 'bar_x_up', 'wbar_x_up']
DIST = ['d_bar', 'd_drk', 'd1_bar', 'd1_drk', 'dd_miss']
DINTER = ['dbar_x_up', 'ddrk_x_up', 'd1bar_x_up', 'd1drk_x_up']


def _group(P):
    if len(P) == 0:
        return 0
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    return int((d < 0.10).sum(1).max())


def encode_row(xy, keep, wards):
    """xy (10, 2) canonical [0, 1]; keep (10,) bool; wards list of (x, y, team)."""
    f = {}
    side = {}
    for t, sl in (('b', slice(0, 5)), ('r', slice(5, 10))):
        P = xy[sl][keep[sl]]
        u = (P[:, 0] - P[:, 1] + 1) / 2
        prog = u if t == 'b' else 1 - u
        enemy = (P[:, 0] > P[:, 1]) if t == 'b' else (P[:, 0] < P[:, 1])
        side[t] = dict(vis=len(P), adv=prog.mean() if len(P) else np.nan, deep=int(enemy.sum()),
                       drk=int((np.hypot(*(P - DRAGON).T) < 0.12).sum()) if len(P) else 0,
                       bar=int((np.hypot(*(P - BARON).T) < 0.12).sum()) if len(P) else 0, group=_group(P))
        for nm, pit in (('bar', BARON), ('drk', DRAGON)):
            dd = np.sort(np.hypot(*(P - pit).T)) if len(P) else np.array([])
            side[t][f'd_{nm}'] = dd[:3].mean() if len(dd) else np.nan
            side[t][f'd1_{nm}'] = dd[0] if len(dd) else np.nan
    b, r = side['b'], side['r']
    miss = 0
    for k in ('d_bar', 'd_drk', 'd1_bar', 'd1_drk'):
        v = r[k] - b[k]
        miss |= int(not np.isfinite(v))
        f[k] = v if np.isfinite(v) else 0.0
    f['dd_miss'] = miss
    f['mm_vis'] = b['vis'] - r['vis']
    f['mm_adv'] = (b['adv'] - r['adv']) if np.isfinite(b['adv']) and np.isfinite(r['adv']) else 0.0
    for k in ('deep', 'drk', 'bar', 'group'):
        f[f'mm_{k}'] = b[k] - r[k]
    W = np.array(wards, float).reshape(-1, 3)
    for k in WD:
        f[k] = 0
    for t, sgn in ((0, 1), (1, -1)):
        w = W[W[:, 2] == t]
        if not len(w):
            continue
        enemy = (w[:, 0] > w[:, 1]) if t == 0 else (w[:, 0] < w[:, 1])
        f['mm_w'] += sgn * len(w)
        f['mm_wdeep'] += sgn * int(enemy.sum())
        f['mm_wdrk'] += sgn * int((np.hypot(*(w[:, :2] - DRAGON).T) < 0.15).sum())
        f['mm_wbar'] += sgn * int((np.hypot(*(w[:, :2] - BARON).T) < 0.15).sum())
    return f


def encode(parsed, thr=0.6):
    """parsed: data/v3/parsed2 rows (detector peaks per team, no identity needed). -> DataFrame of features
    (all-NaN rows where the minimap was not found)."""
    rows = []
    for _, r in parsed.iterrows():
        if not isinstance(r.get('peaks'), str):
            rows.append({})
            continue
        pk = [q for q in json.loads(r.peaks) if q[3] >= thr]
        b = [q[:2] for q in pk if q[2] == 0][:5]
        rd = [q[:2] for q in pk if q[2] == 1][:5]
        xy = np.zeros((10, 2))
        keep = np.zeros(10, bool)
        if b:
            xy[:len(b)], keep[:len(b)] = b, True
        if rd:
            xy[5:5 + len(rd)], keep[5:5 + len(rd)] = rd, True
        rows.append(encode_row(xy, keep, json.loads(r.wards)))
    return pd.DataFrame(rows, index=parsed.index)


def add_interactions(f):
    up_d = f.drake_up + f.elder_up
    f['drk_x_up'] = up_d * f.mm_drk
    f['wdrk_x_up'] = up_d * f.mm_wdrk
    f['bar_x_up'] = f.baron_up * f.mm_bar
    f['wbar_x_up'] = f.baron_up * f.mm_wbar
    if 'd_bar' in f:
        f['dbar_x_up'] = f.baron_up * f.d_bar
        f['ddrk_x_up'] = up_d * f.d_drk
        f['d1bar_x_up'] = f.baron_up * f.d1_bar
        f['d1drk_x_up'] = up_d * f.d1_drk
    return f
