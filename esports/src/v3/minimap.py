"""Minimap parser: broadcast frame -> champion positions (per participant) and ward positions (per team).

Every Riot-produced broadcast in 2024-26 uses the same stylised esports minimap in the bottom-right corner; only its size
(~225-280 px at 1080p) and offset change by league and year. So:
  locate()   multi-scale normalised cross-correlation of a reference minimap against the bottom-right corner. The score
             separates in-game frames (0.62-0.71 on 25 VODs over 11 leagues) from draft / desk / black frames (0.22-0.27).
  canon()    resample the located square to N x N canonical map coordinates: (0, 0) top-left, blue fountain bottom-left.
  champs()   circle-masked Data Dragon icons of the game's 10 champions, matched at the icon scale; greedy one-to-one
             assignment by score, team-ring colour as a check.
  wards()    small team-coloured blobs that are not in the static background (median over many frames) and not on a lane.
All coordinates returned are in [0, 1] map units, x right, y down.
"""
import os, glob
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '../../data')
REF = f'{D}/v3/ref'
ICONS = f'{D}/v3/icons'
N = 256  # canonical map size (px)
CORNER = (540, 1100)  # top-left of the bottom-right search region in a 1920x1080 frame


def _g(x):
    return cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (3, 3), 0)


_T = None


def template():
    global _T
    if _T is None:
        p = f'{REF}/background.png' if os.path.exists(f'{REF}/background.png') else f'{REF}/tmpl0.png'
        _T = _g(cv2.imread(p))
    return _T


def full_frame(img):
    """a saved corner crop (grab.CROP, or the pilot's 768x486 one) or a whole frame -> 1920x1080 frame (zeros elsewhere)."""
    h, w = img.shape[:2]
    if (h, w) == (1080, 1920):
        return img
    if (w, h) in ((360, 360), (768, 486)):
        f = np.zeros((1080, 1920, 3), np.uint8)
        f[1080 - h:, 1920 - w:] = img
        return f
    return cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)


def locate(img, hint=None):
    """-> (score, x, y, size) of the minimap square in a 1920x1080 BGR frame.

    hint=(x, y, size) restricts the search to +-6 px of size and +-12 px of position (one VOD keeps its layout)."""
    img = full_frame(img)
    y0, x0 = CORNER
    R = _g(img[y0:, x0:])
    T = template()
    best = (-1, 0, 0, 0)
    sizes = range(hint[2] - 6, hint[2] + 7) if hint else list(range(190, 300, 3))
    for s in sizes:
        t = cv2.resize(T, (s, s), interpolation=cv2.INTER_AREA)
        if hint:
            hx, hy = hint[0] - x0 - 12, hint[1] - y0 - 12
            sub = R[max(hy, 0):hy + s + 24, max(hx, 0):hx + s + 24]
            if sub.shape[0] < s or sub.shape[1] < s:
                continue
            r = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            loc = (loc[0] + max(hx, 0), loc[1] + max(hy, 0))
        else:
            r = cv2.matchTemplate(R, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (mx, loc[0] + x0, loc[1] + y0, s)
    if not hint and best[0] > 0:  # refine the 3 px size grid
        return locate(img, best[1:]) if best[0] > 0.3 else best
    return best


def canon(img, loc):
    img = full_frame(img)
    _, x, y, s = loc
    return cv2.resize(img[y:y + s, x:x + s], (N, N), interpolation=cv2.INTER_AREA)


_ICON = {}


def icon(champ, d, zoom=1.0):
    """Circle-masked champion icon at diameter d (canonical px). zoom > 1 crops into the square icon first."""
    k = (champ.lower(), d, zoom)
    if k not in _ICON:
        f = {os.path.basename(p)[:-4].lower(): p for p in glob.glob(f'{ICONS}/*.png')}[champ.lower()]
        im = cv2.imread(f)
        h = im.shape[0]
        c = int(h * (1 - 1 / zoom) / 2)
        im = cv2.resize(im[c:h - c, c:h - c], (d, d), interpolation=cv2.INTER_AREA)
        m = np.zeros((d, d), np.uint8)
        cv2.circle(m, (d // 2, d // 2), int(d * 0.40), 255, -1)  # inner disc: skips the team ring
        _ICON[k] = (im, m)
    return _ICON[k]


def ring_colour(cm, x, y, d):
    """mean (b - r) on the icon's ring annulus: > 0 blue team, < 0 red team."""
    yy, xx = np.ogrid[:cm.shape[0], :cm.shape[1]]
    r2 = (xx - x) ** 2 + (yy - y) ** 2
    a = (r2 >= (0.40 * d) ** 2) & (r2 <= (0.55 * d) ** 2)
    px = cm[a].astype(float)
    return (px[:, 0] - px[:, 2]).mean() if len(px) else 0.0


def champs(cm, champions, d=24, zoom=1.0, thr=0.55):
    """cm: canonical BGR map. champions: list of 10 names (participants 1-10, blue 1-5).

    -> array (10, 3): x, y in [0, 1], match score (nan where not found). Greedy: the highest score over all
    (champion, location) pairs is taken first, then a disc of radius 0.6 d around it is suppressed for the others,
    so two icons can overlap by up to ~40 %."""
    S = []
    for ch in champions:
        im, m = icon(ch, d, zoom)
        r = cv2.matchTemplate(cm, im, cv2.TM_CCOEFF_NORMED, mask=m)
        r = np.nan_to_num(r, nan=0, posinf=0, neginf=0)
        S.append(r)
    S = np.stack(S)  # (10, H, W) top-left positions
    out = np.full((10, 3), np.nan)
    taken = np.zeros(10, bool)
    blocked = np.zeros(S.shape[1:], bool)
    yy, xx = np.ogrid[:S.shape[1], :S.shape[2]]
    for _ in range(10):
        Sm = np.where(blocked[None] | taken[:, None, None], -1, S)
        i, y, x = np.unravel_index(np.argmax(Sm), Sm.shape)
        if Sm[i, y, x] < thr:
            break
        taken[i] = True
        out[i] = ((x + d / 2) / N, (y + d / 2) / N, Sm[i, y, x])
        blocked |= (xx - x) ** 2 + (yy - y) ** 2 < (0.6 * d) ** 2
    return out


_BG = None


def background():
    global _BG
    if _BG is None:
        _BG = cv2.imread(f'{REF}/background.png').astype(np.int16)
    return _BG


def team_masks(cm, diff_thr=40):
    """pixels in a team colour that are not part of the static background: (blue, red) bool masks."""
    c = cm.astype(np.int16)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    fg = np.abs(c - background()).sum(2) > diff_thr
    blue = fg & (b - r > 50) & (b > 110)
    red = fg & (r - b > 50) & (r > 110) & (r - g > 40)
    return blue, red


def wards(cm, champ_xy=None, d=24, amin=12, amax=90):
    """-> list of (x, y, team) for small team-coloured blobs off the champion discs. team 0 blue, 1 red.

    Minion dots are ~2-3 px across (area < 12) and wards ~7-9 px, so area in [amin, amax] keeps wards and
    drops minions; blobs within 0.6 d of a detected champion centre are ring fragments and are dropped."""
    out = []
    excl = np.zeros(cm.shape[:2], bool)
    if champ_xy is not None:
        yy, xx = np.ogrid[:N, :N]
        for x, y in champ_xy:
            if np.isfinite(x):
                excl |= (xx - x * N) ** 2 + (yy - y * N) ** 2 < (0.65 * d) ** 2
    for team, m in enumerate(team_masks(cm)):
        m = (m & ~excl).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        for k in range(1, n):
            a, w, h = st[k, cv2.CC_STAT_AREA], st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]
            if amin <= a <= amax and max(w, h) <= 14 and min(w, h) >= 4:
                out.append((cen[k][0] / N, cen[k][1] / N, team))
    return out


def score_maps(cm, names, d, zoom):
    S = []
    for ch in names:
        im, m = icon(ch, d, zoom)
        r = cv2.matchTemplate(cm, im, cv2.TM_CCOEFF_NORMED, mask=m)
        S.append(np.nan_to_num(r, nan=0, posinf=0, neginf=0))
    return np.stack(S)


def champs2(cm, champions, decoys, d, zoom, thr=0.6, margin=0.05):
    """champs() with a likelihood-ratio check: at a candidate location the champion's score must beat the best of
    `decoys` (champions not in this game) by `margin`. A location where some other icon fits about as well is an
    overlap, ring or structure, not this champion. -> (10, 4): x, y, score, margin."""
    S = score_maps(cm, champions, d, zoom)
    Dm = score_maps(cm, decoys, d, zoom).max(0)
    M = S - Dm[None]
    out = np.full((10, 4), np.nan)
    taken = np.zeros(10, bool)
    blocked = np.zeros(S.shape[1:], bool)
    yy, xx = np.ogrid[:S.shape[1], :S.shape[2]]
    ok = (S >= thr) & (M >= margin)
    for _ in range(10):
        Sm = np.where(ok & ~blocked[None] & ~taken[:, None, None], S, -1)
        i, y, x = np.unravel_index(np.argmax(Sm), Sm.shape)
        if Sm[i, y, x] < 0:
            break
        taken[i] = True
        out[i] = ((x + d / 2) / N, (y + d / 2) / N, S[i, y, x], M[i, y, x])
        blocked |= (xx - x) ** 2 + (yy - y) ** 2 < (0.6 * d) ** 2
    return out


def ring_masks(cm):
    """team ring colour masks (no background subtraction: rings are saturated, the map is not)."""
    c = cm.astype(np.int16)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    blue = (b - r > 60) & (b > 140) & (b - g > 10)
    red = (r - b > 60) & (r > 140) & (r - g > 60)
    return blue, red


def ring_candidates(cm, d, cover=0.35, structs=None):
    """-> list of (x, y, team, coverage): local maxima of the share of an annulus (radius ~d/2) in team ring colour.

    A clean icon covers ~70-90 % of its annulus; half-hidden ones in a fight still show >= 35 %."""
    out = []
    k = np.zeros((d + 3, d + 3), np.float32)
    cv2.circle(k, ((d + 3) // 2, (d + 3) // 2), int(round(d / 2)), 1, 2)
    k /= k.sum()
    for team, m in enumerate(ring_masks(cm)):
        if structs is not None:
            m = m & ~structs
        cov = cv2.filter2D(m.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
        mx = cv2.dilate(cov, np.ones((int(d * 0.6), int(d * 0.6)), np.uint8))
        ys, xs = np.nonzero((cov >= mx) & (cov >= cover))
        for x, y in zip(xs, ys):
            out.append((x, y, team, float(cov[y, x])))
    return out


def champs3(cm, champions, d, zoom, cover=0.35, structs=None):
    """ring candidates, then each team's 5 champions assigned to that team's candidates by template score (Hungarian).

    -> (10, 4): x, y in [0, 1], identity score, ring coverage; nan where a champion got no candidate."""
    from scipy.optimize import linear_sum_assignment
    cand = ring_candidates(cm, d, cover, structs)
    out = np.full((10, 4), np.nan)
    h = d // 2
    for team in (0, 1):
        cs = [c for c in cand if c[2] == team]
        if not cs:
            continue
        cs = sorted(cs, key=lambda c: -c[3])[:7]
        S = np.full((5, len(cs)), -1.0)
        for j in range(5):
            im, m = icon(champions[team * 5 + j], d, zoom)
            for k, (x, y, _, _) in enumerate(cs):
                y0, x0 = max(y - h - 2, 0), max(x - h - 2, 0)
                sub = cm[y0:y + h + 3, x0:x + h + 3]
                if sub.shape[0] < d or sub.shape[1] < d:
                    continue
                r = cv2.matchTemplate(sub, im, cv2.TM_CCOEFF_NORMED, mask=m)
                S[j, k] = np.nan_to_num(r, nan=-1).max()
        rows, cols = linear_sum_assignment(-S)
        for j, k in zip(rows, cols):
            if S[j, k] > 0.2:
                x, y, _, cv_ = cs[k]
                out[team * 5 + j] = (x / N, y / N, S[j, k], cv_)
    return out


def fg_cover(cm, bg, d, thr=60):
    """share of an icon-sized disc (radius 0.4 d) whose pixels differ from the game's background: (H-d+1, W-d+1),
    aligned with matchTemplate's top-left indexing."""
    fg = (np.abs(cm.astype(np.int16) - bg.astype(np.int16)).sum(2) > thr).astype(np.float32)
    k = np.zeros((d, d), np.float32)
    cv2.circle(k, (d // 2, d // 2), int(d * 0.4), 1, -1)
    k /= k.sum()
    c = cv2.filter2D(fg, -1, k, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    return c[:cm.shape[0] - d + 1, :cm.shape[1] - d + 1]


def champs4(cm, bg, champions, decoys, d, zoom, thr=0.6, margin=0.05, cover=0.6):
    """champs2() restricted to locations where an icon-sized disc is mostly foreground (differs from the game's
    median background), so an empty patch of map can never hold a champion. -> (10, 5): x, y, score, margin, cover."""
    S = score_maps(cm, champions, d, zoom)
    Dm = score_maps(cm, decoys, d, zoom).max(0) if decoys else np.zeros(S.shape[1:])
    C = fg_cover(cm, bg, d)
    M = S - Dm[None]
    out = np.full((10, 5), np.nan)
    taken = np.zeros(10, bool)
    blocked = np.zeros(S.shape[1:], bool)
    yy, xx = np.ogrid[:S.shape[1], :S.shape[2]]
    ok = (S >= thr) & (M >= margin) & (C[None] >= cover)
    for _ in range(10):
        Sm = np.where(ok & ~blocked[None] & ~taken[:, None, None], S, -1)
        i, y, x = np.unravel_index(np.argmax(Sm), Sm.shape)
        if Sm[i, y, x] < 0:
            break
        taken[i] = True
        out[i] = ((x + d / 2) / N, (y + d / 2) / N, S[i, y, x], M[i, y, x], C[y, x])
        blocked |= (xx - x) ** 2 + (yy - y) ** 2 < (0.6 * d) ** 2
    return out
