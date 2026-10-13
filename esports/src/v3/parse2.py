"""v3 step 5e: parse crops with the trained icon detector (train_det.py) instead of raw template matching.

Per game: locate the minimap on each crop (full search on the first, then a hint), canonical maps, the game's median
background (for wards), icon size/zoom fitted on 6 frames (as parse.py). Per frame: detector heatmaps -> peaks per team
(p >= THR); then identity: Hungarian assignment of the team's *alive* champions (feed currentHealth > 0 at that frame; the
live tracker has the same information) to that team's peaks by masked template score. Peaks beyond the alive count keep
the team but no identity. Wards as parse.py, with detected icon discs excluded.
Output data/v3/parsed2/{gid}.parquet: ts, loc, mm_x, mm_y, mm_size, n_b, n_r (peaks per team), x_i, y_i, p_i, s_i, wards.
usage (from esports/src): python v3/parse2.py IDS_FILE [WORKERS]
"""
import os, sys, glob, json
import cv2, numpy as np, pandas as pd, torch
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minimap as mm
from parse import meta, struct_mask, wards_bg, GRID, LOC_THR
from train_det import Net, to_t, peaks
from validate import feed_alive

D = '../data'
CROPS, OUT = f'{D}/v3/crops', f'{D}/v3/parsed2'
THR = 0.4
_NET = None


def net(dev):
    global _NET
    if _NET is None:
        _NET = Net()
        _NET.load_state_dict(torch.load(f'{D}/v3/ref/det.pt', map_location='cpu'))
        _NET.eval().to(dev)
    return _NET


def identity(cm, pk, champs, alive, d, z):
    """pk: [(x, y, team, p)] canonical px. -> (10, 4) x, y, p, s per participant (nan if unassigned)."""
    out = np.full((10, 4), np.nan)
    h = d // 2
    for team in (0, 1):
        P = [q for q in pk if q[2] == team]
        js = [j for j in range(5) if alive[team * 5 + j]]
        if not P or not js:
            continue
        S = np.full((len(js), len(P)), -1.0)
        for a, j in enumerate(js):
            im, m = mm.icon(champs[team * 5 + j], d, z)
            for b, (x, y, _, _) in enumerate(P):
                x0, y0 = int(round(x)) - h - 2, int(round(y)) - h - 2
                sub = cm[max(y0, 0):y0 + d + 4, max(x0, 0):x0 + d + 4]
                if sub.shape[0] < d or sub.shape[1] < d:
                    continue
                r = cv2.matchTemplate(sub, im, cv2.TM_CCOEFF_NORMED, mask=m)
                S[a, b] = np.nan_to_num(r, nan=-1).max()
        ra, cb = linear_sum_assignment(-S)
        for a, b in zip(ra, cb):
            x, y, _, p = P[b]
            out[team * 5 + js[a]] = (x / mm.N, y / mm.N, p, S[a, b])
    return out


def game(gid, dev='cpu'):
    out = f'{OUT}/{gid}.parquet'
    if os.path.exists(out):
        return gid, 'skip'
    fs = sorted(glob.glob(f'{CROPS}/{gid}/[0-9]*.png'))
    if not fs or not os.path.exists(f'{D}/frames60/{gid}.meta.json'):
        return gid, 'nocrops'
    champs = meta(gid)
    al = feed_alive(gid).set_index('sec')
    imgs = [cv2.imread(f) for f in fs]
    hint, locs = None, []
    for im in imgs:
        l = mm.locate(im, hint[1:]) if hint else mm.locate(im)
        if hint and l[0] < LOC_THR:
            l = mm.locate(im)
        if l[0] >= LOC_THR and hint is None:
            hint = l
        locs.append(l)
    good = [i for i, l in enumerate(locs) if l[0] >= LOC_THR]
    secs = [int(os.path.basename(f)[:-4]) for f in fs]
    rows = [dict(game_id=gid, ts=pd.Timestamp(s, unit='s', tz='UTC'), loc=l[0], mm_x=l[1], mm_y=l[2], mm_size=l[3])
            for s, l in zip(secs, locs)]
    if len(good) < 5:
        pd.DataFrame(rows).to_parquet(out)
        return gid, 'nominimap'
    C = {i: mm.canon(imgs[i], locs[i]) for i in good}
    bg = np.median(np.stack([C[i] for i in good]), 0)
    smask = struct_mask(bg)
    sample = good[::max(1, len(good) // 6)][:6]
    fit = {k: np.mean([np.sort(mm.score_maps(C[i], champs, *k).max((1, 2)))[-5:].mean() for i in sample]) for k in GRID}
    d, z = max(fit, key=fit.get)
    nt = net(dev)
    with torch.no_grad():
        X = torch.stack([to_t(C[i]) for i in good]).to(dev)
        H = torch.cat([torch.sigmoid(nt(X[k:k + 16])) for k in range(0, len(X), 16)]).cpu().numpy()
    for k, i in enumerate(good):
        pk = peaks(H[k], THR)
        s = secs[i]
        alive = [bool(al.loc[s, f'p{j}_currentHealth'] > 0) if s in al.index else True for j in range(1, 11)]
        o = identity(C[i], pk, champs, alive, d, z)
        w = wards_bg(C[i], bg, smask, [(x / mm.N, y / mm.N) for x, y, _, _ in pk], d)
        r = rows[i]
        r.update(icon_d=d, icon_z=z, n_b=sum(q[2] == 0 for q in pk), n_r=sum(q[2] == 1 for q in pk),
                 peaks=json.dumps([(round(x / mm.N, 4), round(y / mm.N, 4), t, round(p, 3)) for x, y, t, p in pk]),
                 wards=json.dumps([(round(x, 4), round(y, 4), t) for x, y, t in w]))
        for j in range(10):
            r[f'x{j + 1}'], r[f'y{j + 1}'], r[f'p{j + 1}'], r[f's{j + 1}'] = o[j]
    pd.DataFrame(rows).to_parquet(out)
    return gid, 'ok'


def _worker(gids):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    st = {}
    for g in gids:
        try:
            s = game(g)[1]
        except Exception as e:
            s = 'err:' + type(e).__name__
        st[s] = st.get(s, 0) + 1
    return st


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(OUT, exist_ok=True)
    ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    ids = [g for g in ids if os.path.exists(f'{CROPS}/{g}/done.json') and not os.path.exists(f'{OUT}/{g}.parquet')]
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    chunks = [ids[k::W] for k in range(W)]
    tot = {}
    with ProcessPoolExecutor(W) as ex:
        for st in ex.map(_worker, chunks):
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
    print(tot, flush=True)
