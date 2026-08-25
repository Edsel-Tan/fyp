#!/usr/bin/env python3
"""Independent pandas recomputation of representative Alpha158 features,
checked against the vectorised numpy implementation in data.py."""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D

rng = np.random.default_rng(7)
T, N, d = 400, 4, 20
c = np.cumprod(1 + rng.normal(0, 0.02, (T, N)), 0) * 50
o = c * (1 + rng.normal(0, 0.005, (T, N)))
h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.004, (T, N))))
l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.004, (T, N))))
v = rng.lognormal(12, 0.4, (T, N))
X, names = D.alpha158(o, h, l, c, v, (h + l + c) / 3)
idx = {n: i for i, n in enumerate(names)}

C = pd.DataFrame(c); H = pd.DataFrame(h); L = pd.DataFrame(l); V = pd.DataFrame(v)
ref = {
    "KMID":    (pd.DataFrame(c - o) / pd.DataFrame(o)),
    f"MA{d}":  C.rolling(d).mean() / C,
    f"STD{d}": C.rolling(d).std() / C,
    f"ROC{d}": C.shift(d) / C,
    f"MAX{d}": H.rolling(d).max() / C,
    f"MIN{d}": L.rolling(d).min() / C,
    f"QTLU{d}": C.rolling(d).quantile(0.8) / C,
    f"RANK{d}": C.rolling(d).apply(lambda w: (w <= w[-1]).sum() / d, raw=True),
    f"RSV{d}": (C - L.rolling(d).min()) / (H.rolling(d).max() - L.rolling(d).min() + 1e-12),
    f"CNTP{d}": (C > C.shift(1)).rolling(d).mean(),
    f"CORR{d}": C.rolling(d).corr(np.log(V + 1)),
    f"VMA{d}": V.rolling(d).mean() / V,
    f"IMAX{d}": H.rolling(d).apply(lambda w: np.argmax(w), raw=True) / d,
    f"SUMP{d}": (C.diff().clip(lower=0).rolling(d).sum()
                 / (C.diff().abs().rolling(d).sum() + 1e-12)),
}
# BETA/RSQR/RESI: rolling OLS on t = 0..d-1
tt = np.arange(d, dtype=float)
def ols(w, what):
    s, i = np.polyfit(tt, w, 1)
    if what == "beta":  return s
    fit = s * tt + i
    if what == "resi":  return w[-1] - fit[-1]
    ss = ((w - w.mean()) ** 2).sum()
    return 1 - ((w - fit) ** 2).sum() / ss if ss > 1e-12 else np.nan
ref[f"BETA{d}"] = C.rolling(d).apply(lambda w: ols(w, "beta"), raw=True) / C
ref[f"RSQR{d}"] = C.rolling(d).apply(lambda w: ols(w, "rsqr"), raw=True)
ref[f"RESI{d}"] = C.rolling(d).apply(lambda w: ols(w, "resi"), raw=True) / C

fails = 0
print(f"{'feature':<10}{'max abs diff':>16}{'checked':>10}")
for name, r in ref.items():
    mine = X[:, :, idx[name]].astype(np.float64)
    theirs = r.to_numpy(np.float64)
    m = np.isfinite(mine) & np.isfinite(theirs)
    diff = np.abs(mine[m] - theirs[m]).max() if m.any() else np.nan
    scale = max(1.0, np.abs(theirs[m]).max() if m.any() else 1.0)
    ok = diff <= 1e-6 * scale
    fails += not ok
    print(f"{name:<10}{diff:>16.3e}{m.sum():>10}  {'ok' if ok else 'MISMATCH'}")
print(f"\n{len(ref)-fails}/{len(ref)} features match; total feature count = {len(names)}")
sys.exit(1 if fails else 0)
