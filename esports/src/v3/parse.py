"""v3 step 5: parse every extracted crop of a game into champion and ward positions.

Per game (data/v3/crops/{gid}): locate the minimap on each crop (full search once, then a hint, since a VOD keeps its
layout), resample to canonical, build the game's own background (median over its frames: icons move, the map does not,
which also absorbs the broadcast's colour grading and the ±1 px alignment error), choose the icon size/zoom that best fits
this broadcast on 6 frames, then champs2() and wards().
Output data/v3/parsed/{gid}.parquet: one row per crop with ts, loc score, x_i, y_i, s_i (i = participant 1..10), and
wards as a JSON list [(x, y, team)].
usage (from esports/src): python v3/parse.py [IDS_FILE] [WORKERS]
"""
import os, sys, glob, json, random
import cv2, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minimap as mm
cv2.setNumThreads(1)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
CROPS, OUT = f'{D}/v3/crops', f'{D}/v3/parsed'
ALL = [l.strip() for l in open(f'{D}/v3/icons/ids.txt')]
GRID = [(d, z) for d in (18, 20, 22, 24, 26) for z in (1.0, 1.15)]
LOC_THR = 0.45


def meta(gid):
    m = json.load(open(f'{D}/frames60/{gid}.meta.json'))
    return [p['championId'] for s in ('blueTeamMetadata', 'redTeamMetadata') for p in m[s]['participantMetadata']]


def struct_mask(bg):
    """structures (team-coloured blobs in the game's own background), dilated: wards are not searched there."""
    b, r = mm.team_masks(bg.astype(np.uint8), diff_thr=-1)
    return cv2.dilate((b | r).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def wards_bg(cm, bg, smask, champ_xy, d):
    old = mm._BG
    mm._BG = bg.astype(np.int16)
    try:
        w = mm.wards(cm, champ_xy, d)
    finally:
        mm._BG = old
    return [(x, y, t) for x, y, t in w if not smask[min(int(y * mm.N), mm.N - 1), min(int(x * mm.N), mm.N - 1)]]


def game(gid):
    out = f'{OUT}/{gid}.parquet'
    if os.path.exists(out):
        return gid, 'skip'
    fs = sorted(glob.glob(f'{CROPS}/{gid}/[0-9]*.png'))
    if not fs:
        return gid, 'nocrops'
    champions = meta(gid)
    decoys = random.Random(int(gid) % 9973).sample([c for c in ALL if c.lower() not in {x.lower() for x in champions}], 15)
    imgs = [cv2.imread(f) for f in fs]
    hint, locs = None, []
    for im in imgs:
        l = mm.locate(im, hint[1:] if hint else None)
        if hint and l[0] < LOC_THR:  # layout may change (replay / different scene): full search
            l = mm.locate(im)
        if l[0] >= LOC_THR and hint is None:
            hint = l
        locs.append(l)
    good = [i for i, l in enumerate(locs) if l[0] >= LOC_THR]
    rows = [dict(game_id=gid, ts=pd.Timestamp(int(os.path.basename(f)[:-4]), unit='s', tz='UTC'), loc=locs[i][0],
                 mm_size=locs[i][3]) for i, f in enumerate(fs)]
    if len(good) < 5:
        pd.DataFrame(rows).to_parquet(out)
        return gid, 'nominimap'
    C = {i: mm.canon(imgs[i], locs[i]) for i in good}
    bg = np.median(np.stack([C[i] for i in good]), 0)
    smask = struct_mask(bg)
    # icon size / zoom for this broadcast: best mean of the top-5 champion scores on up to 6 frames
    sample = good[::max(1, len(good) // 6)][:6]
    fit = {}
    for d, z in GRID:
        fit[(d, z)] = np.mean([np.sort(mm.score_maps(C[i], champions, d, z).max((1, 2)))[-5:].mean() for i in sample])
    d, z = max(fit, key=fit.get)
    for i in good:
        o = mm.champs4(C[i], bg, champions, decoys, d, z, thr=0.5, margin=0.0, cover=0.4)
        w = wards_bg(C[i], bg, smask, o[:, :2], d)
        r = rows[i]
        r.update(icon_d=d, icon_z=z, icon_fit=fit[(d, z)], wards=json.dumps([(round(x, 4), round(y, 4), t) for x, y, t in w]))
        for k in range(10):
            r[f'x{k + 1}'], r[f'y{k + 1}'], r[f's{k + 1}'], r[f'm{k + 1}'], r[f'c{k + 1}'] = o[k]
    pd.DataFrame(rows).to_parquet(out)
    return gid, 'ok'


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    ids = [l.strip() for l in open(sys.argv[1])] if len(sys.argv) > 1 else \
        [os.path.basename(p) for p in glob.glob(f'{CROPS}/*') if os.path.exists(f'{p}/done.json')]
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    st = {}
    with ProcessPoolExecutor(W) as ex:
        for g, s in ex.map(game, ids):
            st[s] = st.get(s, 0) + 1
    print(st)
