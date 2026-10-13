"""v3 step 5c: synthetic broadcast minimaps with exact labels, for training the icon detector.

A sample = one game's clean canonical background (median of its frames) + augmentations that real frames show:
  fog-of-war: random soft-edged regions darkened/lightened (the broadcast shades by vision)
  camera box: the white rectangle;  minion chains and ward blobs: small blue/red dots (distractors)
  icons: 0-5 per team, Data Dragon art zoomed and circle-cropped, with a team ring (colours sampled around the
         medians measured on real high-confidence icons), dark outline, overlaps allowed (fights)
then the whole map is upscaled to a random broadcast size (225-280 px), blurred, JPEG-compressed at q 35-85 and
resampled back to N x N, which is roughly what the YouTube encode does.
Label: (x, y, team) per icon, rendered by the trainer as Gaussian heatmaps (CenterNet style).
"""
import os, glob, random
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '../../data')
N = 256
ICONS = sorted(glob.glob(f'{D}/v3/icons/*.png'))
RING = np.array([[215, 135, 45], [55, 50, 205]], float)  # BGR blue / red team rings, read off real icons (real_icons.png)


class Synth:
    def __init__(self, backgrounds, seed=0):
        self.bg = backgrounds
        self.rng = np.random.default_rng(seed)
        self.icons = [cv2.imread(p) for p in ICONS]

    def icon(self, d, team):
        r = self.rng
        im = self.icons[r.integers(len(self.icons))]
        h = im.shape[0]
        z = r.uniform(1.0, 1.3)
        c = int(h * (1 - 1 / z) / 2)
        S = 4 * d  # draw at 4x then downscale: smooth edges
        im = cv2.resize(im[c:h - c, c:h - c], (S, S), interpolation=cv2.INTER_AREA)
        out = np.zeros((S, S, 3), np.float32)
        a = np.zeros((S, S), np.float32)
        yy, xx = np.mgrid[:S, :S]
        rr = np.hypot(xx - S / 2, yy - S / 2) / S
        inner, ring_w = r.uniform(0.38, 0.43), r.uniform(0.04, 0.09)
        col = np.clip(RING[team] * r.uniform(0.75, 1.15) + r.normal(0, 12, 3), 0, 255)
        out[rr < inner] = im[rr < inner] * r.uniform(0.6, 1.0)
        m = (rr >= inner) & (rr < inner + ring_w)
        out[m] = col
        o = (rr >= inner + ring_w) & (rr < inner + ring_w + 0.03)
        out[o] = 15
        a[rr < inner + ring_w + 0.03] = 1
        if r.random() < 0.12:  # highlighted icon: bright outer glow in team colour
            g = (rr >= inner + ring_w) & (rr < 0.5)
            out[g] = np.clip(col * 1.3 + 40, 0, 255)
            a[g] = 0.8
        k = int(d * (2 * (inner + ring_w + 0.03)) + 1)
        k = max(k, 5)
        out = cv2.resize(out, (d, d), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, (d, d), interpolation=cv2.INTER_AREA)
        return out, a

    def sample(self, n_max=5):
        r = self.rng
        img = self.bg[r.integers(len(self.bg))].astype(np.float32).copy()
        # fog: a few soft blobs, multiplicative
        fog = np.ones((N, N), np.float32)
        for _ in range(r.integers(0, 7)):
            m = np.zeros((N, N), np.float32)
            cv2.ellipse(m, (int(r.uniform(0, N)), int(r.uniform(0, N))), (int(r.uniform(15, 70)), int(r.uniform(15, 70))),
                        r.uniform(0, 180), 0, 360, 1, -1)
            m = cv2.GaussianBlur(m, (0, 0), r.uniform(3, 10))
            fog *= 1 + m * r.uniform(-0.45, 0.25)
        img *= fog[..., None]
        # minion chains / ward-like dots (distractors)
        for _ in range(r.integers(0, 10)):
            team = r.integers(2)
            x, y = r.uniform(5, N - 5, 2)
            ang = r.uniform(0, np.pi)
            for k in range(r.integers(1, 7)):
                cv2.circle(img, (int(x + k * 3.5 * np.cos(ang)), int(y + k * 3.5 * np.sin(ang))), int(r.integers(1, 3)),
                           tuple(float(v) for v in RING[team] * r.uniform(0.7, 1.1)), -1)
        for _ in range(r.integers(0, 10)):
            team = r.integers(2)
            x, y = r.uniform(5, N - 5, 2)
            cv2.circle(img, (int(x), int(y)), int(r.integers(2, 5)), tuple(float(v) for v in RING[team] * r.uniform(0.6, 1.1)), -1)
        # icons
        d = int(r.integers(17, 28))
        labels = []
        cluster = r.random() < 0.4
        cx, cy = r.uniform(30, N - 30, 2)
        for team in (0, 1):
            for _ in range(r.integers(0, n_max + 1)):
                if cluster and r.random() < 0.6:
                    x, y = cx + r.normal(0, d * 0.8), cy + r.normal(0, d * 0.8)
                else:
                    x, y = r.uniform(d / 2, N - d / 2, 2)
                x, y = float(np.clip(x, d / 2, N - d / 2 - 1)), float(np.clip(y, d / 2, N - d / 2 - 1))
                labels.append((x, y, team))
        order = r.permutation(len(labels))
        for k in order:
            x, y, team = labels[k]
            ic, a = self.icon(d, team)
            x0, y0 = int(round(x - d / 2)), int(round(y - d / 2))
            sub = img[y0:y0 + d, x0:x0 + d]
            hh, ww = sub.shape[:2]
            sub[:] = sub * (1 - a[:hh, :ww, None]) + ic[:hh, :ww] * a[:hh, :ww, None]
        # camera rectangle
        if r.random() < 0.8:
            x, y = r.uniform(0, N - 60, 2)
            cv2.rectangle(img, (int(x), int(y)), (int(x + r.uniform(50, 110)), int(y + r.uniform(35, 70))), (235, 235, 235), 1 + int(r.random() < 0.3))
        # broadcast encode
        img = np.clip(img, 0, 255).astype(np.uint8)
        s = int(r.integers(225, 281))
        big = cv2.resize(img, (s, s), interpolation=cv2.INTER_CUBIC)
        big = cv2.GaussianBlur(big, (0, 0), r.uniform(0.3, 1.0))
        ok, enc = cv2.imencode('.jpg', big, [cv2.IMWRITE_JPEG_QUALITY, int(r.integers(35, 86))])
        big = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        img = cv2.resize(big, (N, N), interpolation=cv2.INTER_AREA)
        return img, labels, d
