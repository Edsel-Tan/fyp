"""v3 step 4: pull the minimap corner of the broadcast at every 60 s feed frame of a game.

usage (from esports/src): python v3/extract.py IDS_FILE [WORKERS]
For each game: frames60 timestamps in state in_game (the clock from ingame_data), VOD second = ts - firstFrameTime,
one ffmpeg seek per frame on the resolved DASH url (grab.frame), the bottom-right corner (grab.CROP) of the frame scaled to 1080p
saved to data/v3/crops/{gid}/{unix_ts}.png. A game whose first in-game-looking crop never locates a minimap
(sync broken, VOD of the wrong game, dead link) is tried on the next vod candidate, then marked bad.
Resumable: a game with data/v3/crops/{gid}/done.json is skipped.
"""
import os, sys, json, time
import cv2, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ingame_data import one as frames_one
from grab import stream_url, frame
from minimap import locate

D = '../data'
OUT = f'{D}/v3/crops'
VODS = pd.read_parquet(f'{D}/v3/vods.parquet')


def probe(url, ff, ts, d):
    """grab 3 frames spread over the game; ok if >= 2 locate a minimap."""
    k = [len(ts) // 4, len(ts) // 2, 3 * len(ts) // 4]
    good = 0
    for i in k:
        p = f'{d}/probe.png'
        if frame(url, (ts[i] - ff).total_seconds(), p):
            good += locate(cv2.imread(p))[0] > 0.45
    return good >= 2


def game(gid):
    d = f'{OUT}/{gid}'
    if os.path.exists(f'{d}/done.json'):
        st = json.load(open(f'{d}/done.json'))
        # 'bad' before 2026-10-05 07:00 was mostly YouTube's bot block (HTTP 429), not the VOD: retry unless a probe really failed
        if st.get('status') != 'bad' or st.get('reason') == 'probe':
            return gid, 'skip'
    os.makedirs(d, exist_ok=True)
    o = frames_one(gid)
    if o is None:
        json.dump(dict(status='noframes'), open(f'{d}/done.json', 'w'))
        return gid, 'noframes'
    o = o[o.state == 'in_game']
    ts = list(o.ts)
    v = VODS[VODS.game_id == str(gid)].sort_values('rank')
    reasons = []
    for _, r in v.iterrows():
        url = stream_url(r.yt)
        if not url:
            reasons.append('nourl')
            continue
        ff = pd.Timestamp(r.first_frame)
        if not probe(url, ff, ts, d):
            reasons.append('probe')
            continue
        ok = 0
        with ThreadPoolExecutor(3) as ex:
            res = ex.map(lambda t: frame(url, (t - ff).total_seconds(), f'{d}/{int(t.timestamp())}.png'), ts)
            ok = sum(res)
        os.path.exists(f'{d}/probe.png') and os.remove(f'{d}/probe.png')
        json.dump(dict(status='ok', yt=r.yt, first_frame=r.first_frame, n=len(ts), ok=int(ok)), open(f'{d}/done.json', 'w'))
        return gid, 'ok'
    os.path.exists(f'{d}/probe.png') and os.remove(f'{d}/probe.png')
    reason = 'probe' if 'probe' in reasons else 'nourl'
    json.dump(dict(status='bad', reason=reason, tried=v.yt.tolist()), open(f'{d}/done.json', 'w'))
    if reason == 'nourl' and len(v):  # every candidate failed to resolve: likely blocked, slow down
        time.sleep(60)
    return gid, 'bad'


if __name__ == '__main__':
    ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    st, t0 = {}, time.time()
    with ThreadPoolExecutor(W) as ex:
        for i, (g, s) in enumerate(ex.map(game, ids)):
            st[s] = st.get(s, 0) + 1
            print(f'{i + 1}/{len(ids)} {g} {s} {st} {(time.time() - t0) / 60:.1f} min', flush=True)
