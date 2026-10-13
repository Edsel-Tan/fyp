"""Grab broadcast frames for a game at given feed timestamps: resolve the YouTube stream once, then one ffmpeg seek per frame.

VOD second = feed wall time - firstFrameTime (vods.parquet). Only the minimap corner (CROP) is kept. Usage as a module: grab(gid, [ts...], outdir).
"""
import os, subprocess, sys
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
PY = sys.executable
COOKIES = os.path.expanduser('~/fyp/cookies.txt')  # user's YouTube cookies (2026-10-05) against the bot block; never print or copy
CROP = (1560, 720, 360, 360)  # x, y, w, h in the 1080p frame: holds every minimap seen (x >= 1636, y >= 795, size <= 277)
FMT = 'bv*[height=1080][protocol=https][vcodec^=avc1]/bv*[height=1080][protocol=https]/bv*[height<=1080][protocol=https]'  # DASH seeks by byte range


def stream_url(yt):
    ck = ['--cookies', COOKIES] if os.path.exists(COOKIES) else []
    r = subprocess.run([PY, '-m', 'yt_dlp', *ck, '-f', FMT, '-g', f'https://www.youtube.com/watch?v={yt}'],
                       capture_output=True, text=True, timeout=120)
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None


def frame(url, sec, out, crop=True, scale=None):
    vf = []
    if scale:
        vf.append(f'scale={scale}')
    if crop:  # bottom-right corner of the (scaled-to-1080p) frame
        vf += ['scale=1920:1080', f'crop={CROP[2]}:{CROP[3]}:{CROP[0]}:{CROP[1]}']
    cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{sec:.2f}', '-i', url, '-frames:v', '1'] + \
          (['-vf', ','.join(vf)] if vf else []) + [out]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    return r.returncode == 0 and os.path.exists(out)


def grab(gid, ts, outdir, crop=True, workers=4, vods=None):
    """ts: feed timestamps (pd.Timestamp, UTC). Returns list of written paths (None where it failed)."""
    vods = pd.read_parquet(f'{D}/v3/vods.parquet') if vods is None else vods
    v = vods[vods.game_id == str(gid)].sort_values('rank')
    os.makedirs(outdir, exist_ok=True)
    for _, r in v.iterrows():
        url = stream_url(r.yt)
        if not url:
            continue
        ff = pd.Timestamp(r.first_frame)
        jobs = [(url, (t - ff).total_seconds(), f'{outdir}/{gid}_{int(t.timestamp())}.png') for t in ts]
        with ThreadPoolExecutor(workers) as ex:
            ok = list(ex.map(lambda j: frame(*j, crop=crop) and j[2], jobs))
        return ok, r.yt
    return [None] * len(ts), None
