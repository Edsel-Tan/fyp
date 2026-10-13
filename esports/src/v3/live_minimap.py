"""v3 live: screenshot the minimap of the live broadcast and turn it into v3's minimap features.

Stream: getLive / getEventDetails `streams` (provider youtube | twitch, parameter = video id | channel), resolved to an HLS
media playlist with yt-dlp (1080p, else the best <= 1080p). Every PERIOD s the newest segment is downloaded and its last
frame decoded; the frame's wall time is the segment's EXT-X-PROGRAM-DATE-TIME + its duration. That clock is what the VOD
sync used (feed ts = broadcast wall time, checked on the in-game timer), so a snapshot is matched to the feed frame with the
nearest timestamp. Parsing is the offline pipeline: locate (hint after the first hit) -> canonical -> detector peaks per
team -> wards against a running median background of the last 20 snapshots -> features.encode_row.

MinimapStream(provider, parameter).start() runs a thread; .at(feed_ts, tol=20) returns (features dict, age_s) or None.
"""
import os, sys, re, time, threading, subprocess, tempfile, datetime as dt, json
import numpy as np, cv2, requests, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minimap as mm
import features as FT
from train_det import Net, to_t, peaks
from parse import struct_mask

PERIOD = 10
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
FMT = 'best[height<=1080]/bv*[height<=1080]'


def resolve(provider, parameter):
    url = {'youtube': f'https://www.youtube.com/watch?v={parameter}', 'twitch': f'https://www.twitch.tv/{parameter}'}.get(provider)
    if not url:
        return None
    r = subprocess.run([sys.executable, '-m', 'yt_dlp', '-f', FMT, '-g', url], capture_output=True, text=True, timeout=90)
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None


def last_segment(m3u8):
    """-> (segment url, wall time of its last frame) from an HLS media playlist."""
    txt = requests.get(m3u8, timeout=10).text
    pdt, dur, seg = None, None, None
    for line in txt.splitlines():
        if line.startswith('#EXT-X-PROGRAM-DATE-TIME:'):
            pdt = dt.datetime.fromisoformat(line.split(':', 1)[1].strip().replace('Z', '+00:00'))
        elif line.startswith('#EXTINF:'):
            dur = float(line[8:].split(',')[0])
        elif line and not line.startswith('#'):
            seg = (requests.compat.urljoin(m3u8, line), pdt, dur)
            if pdt is not None and dur is not None:
                pdt = pdt + dt.timedelta(seconds=dur)  # PDT tags may be sparse: carry forward
    if not seg:
        return None, None
    url, p, d = seg
    return url, (p + dt.timedelta(seconds=d or 0)) if p else None


def last_frame(seg_url):
    with tempfile.TemporaryDirectory() as td:
        f = f'{td}/s.ts'
        open(f, 'wb').write(requests.get(seg_url, timeout=15).content)
        out = f'{td}/f.png'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', f, '-vf', 'scale=1920:1080', '-update', '1', out],  # decode all, keep the last (seeking to the end breaks h264 references)
                       capture_output=True, timeout=30)
        return cv2.imread(out) if os.path.exists(out) else None


class MinimapStream:
    def __init__(self, provider, parameter, log=None):
        self.provider, self.parameter = provider, parameter
        self.buf, self.cms = [], []
        self.hint, self.url, self.stop = None, None, threading.Event()
        self.log = log or (lambda *a: None)
        self.net = Net()
        self.net.load_state_dict(torch.load(f'{D}/v3/ref/det.pt', map_location='cpu'))
        self.net.eval()

    def parse(self, img):
        l = mm.locate(img, self.hint[1:]) if self.hint else mm.locate(img)
        if self.hint and l[0] < 0.45:
            l = mm.locate(img)
        if l[0] < 0.45:
            return None
        self.hint = l
        cm = mm.canon(img, l)
        self.cms = (self.cms + [cm])[-20:]
        with torch.no_grad():
            hm = torch.sigmoid(self.net(to_t(cm)[None]))[0].numpy()
        pk = peaks(hm, 0.4)
        wards = []
        if len(self.cms) >= 5:
            bg = np.median(np.stack(self.cms), 0)
            old = mm._BG
            mm._BG = bg.astype(np.int16)
            try:
                w = mm.wards(cm, [(x / mm.N, y / mm.N) for x, y, _, _ in pk], 22)
            finally:
                mm._BG = old
            sm = struct_mask(bg)
            wards = [q for q in w if not sm[min(int(q[1] * mm.N), mm.N - 1), min(int(q[0] * mm.N), mm.N - 1)]]
        b = [(x / mm.N, y / mm.N) for x, y, t, _ in pk if t == 0][:5]
        r = [(x / mm.N, y / mm.N) for x, y, t, _ in pk if t == 1][:5]
        xy, keep = np.zeros((10, 2)), np.zeros(10, bool)
        if b:
            xy[:len(b)], keep[:len(b)] = b, True
        if r:
            xy[5:5 + len(r)], keep[5:5 + len(r)] = r, True
        f = FT.encode_row(xy, keep, wards)
        f['peaks'] = json.dumps([(round(x / mm.N, 4), round(y / mm.N, 4), t, round(p, 3)) for x, y, t, p in pk])
        return f

    def loop(self):
        torch.set_num_threads(2)
        while not self.stop.is_set():
            t0 = time.time()
            try:
                if not self.url:
                    self.url = resolve(self.provider, self.parameter)
                seg, wall = last_segment(self.url) if self.url else (None, None)
                if seg and wall:
                    img = last_frame(seg)
                    f = self.parse(img) if img is not None else None
                    if f is not None:
                        self.buf = (self.buf + [(wall, f)])[-120:]
            except Exception as e:  # url expiry, network: re-resolve next time
                self.log(f'minimap {self.parameter}: {type(e).__name__} {e}')
                self.url = None
            self.stop.wait(max(1, PERIOD - (time.time() - t0)))

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def at(self, feed_ts, tol=20):
        if not self.buf:
            return None
        k = min(range(len(self.buf)), key=lambda i: abs((self.buf[i][0] - feed_ts).total_seconds()))
        age = (feed_ts - self.buf[k][0]).total_seconds()
        return (self.buf[k][1], age) if abs(age) <= tol else None
