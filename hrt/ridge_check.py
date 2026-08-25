#!/usr/bin/env python3
"""Is the Transformer the limiting factor, or is the signal itself just weak?

Fits a cross-sectionally standardised ridge on the same Alpha158 panel and the
same causal label, so the HLC's forecast quality can be attributed to the signal
rather than to the choice of architecture.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast import TRAIN, VALID, TEST, split_idx, build_labels, robust_zscore, cs_zscore, daily_ic

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

z = np.load(os.path.join(ART, "panel.npz"), allow_pickle=True)
X, dates, op = z["X"], z["dates"].astype(str), z["open"].astype(np.float64)
T, N, F = X.shape
tr, va, te = (split_idx(dates, *P) for P in (TRAIN, VALID, TEST))

Xn, _, _ = robust_zscore(X.astype(np.float64), tr)
lab = cs_zscore(build_labels(op, "causal"))
ok = np.isfinite(lab)

def flat(idx):
    t, n = np.meshgrid(idx, np.arange(N), indexing="ij")
    t, n = t.ravel(), n.ravel()
    m = ok[t, n]
    return Xn[t[m], n[m]], lab[t[m], n[m]], t[m]

Xtr, ytr, _ = flat(tr)
print(f"train rows {len(ytr):,}  features {F}")
G = Xtr.T @ Xtr
b = Xtr.T @ ytr
print(f"\n{'alpha':>10}{'valid IC':>12}{'valid ICIR':>12}{'test IC':>12}")
best = (None, -9e9)
for a in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
    w = np.linalg.solve(G + a * np.eye(F), b)
    out = []
    for name, idx in (("valid", va), ("test", te)):
        Xs, ys, ts = flat(idx)
        ic, s = daily_ic(Xs @ w, ys, ts)
        out.append((ic, ic / max(s, 1e-9)))
    print(f"{a:>10.0e}{out[0][0]:>+12.4f}{out[0][1]:>+12.3f}{out[1][0]:>+12.4f}")
    if out[0][0] > best[1]:
        best = (a, out[0][0], out[1][0])
print(f"\nbest ridge by validation IC: alpha={best[0]:.0e}  valid {best[1]:+.4f}  test {best[2]:+.4f}")
print("Transformer (same label/splits):        valid +0.0156  test +0.0118")
