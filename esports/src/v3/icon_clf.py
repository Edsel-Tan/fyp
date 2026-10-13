"""v3 step 5b: a learned check on champion candidates, with the feed as the labeller.

Template matching alone puts champions on empty map: the broadcast minimap has fog-of-war shading that changes every frame,
and some broadcasts draw only a few icons. But the feed knows who is dead, and a dead champion is not on the minimap, so every
candidate for a dead champion is a false positive: a free, hard negative. Candidates for alive champions are mostly true
(and partly false), so P(alive | patch) learnt on them ranks candidates by how icon-like they are.
Features per candidate: the template score, foreground cover, ring colour stats on the annulus, and a 16x16 grey+HSV patch.
Model: HistGradientBoosting, game-grouped 5-fold CV; reports recall vs dead-FP at thresholds; saves data/v3/ref/icon_clf.pkl.
"""
import os, sys, glob, json, pickle
import cv2, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import minimap as mm

D = '../data'


def patch_feats(cm, x, y, d):
    """x, y canonical px centre; -> feature vector."""
    h = int(round(d * 0.6))
    pad = cv2.copyMakeBorder(cm, h, h, h, h, cv2.BORDER_CONSTANT)
    p = pad[int(y):int(y) + 2 * h, int(x):int(x) + 2 * h]
    p = cv2.resize(p, (16, 16), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(p, cv2.COLOR_BGR2HSV).astype(np.float32)
    g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float32)
    yy, xx = np.mgrid[:16, :16]
    rr = np.hypot(xx - 7.5, yy - 7.5) / 16 * (2 * h) / d  # radius in icon diameters
    ring = (rr >= 0.38) & (rr <= 0.55)
    inner = rr < 0.38
    outer = rr > 0.6
    q = p.astype(np.float32)
    f = [q[ring][:, 0].mean() - q[ring][:, 2].mean(), hsv[ring][:, 1].mean(), hsv[ring][:, 2].mean(),
         hsv[inner][:, 1].mean(), hsv[inner][:, 2].mean(), g[inner].std(), g[outer].std(),
         g[inner].mean() - g[outer].mean(), np.abs(g[ring].mean() - g[outer].mean())]
    return np.r_[f, g.ravel() / 255, hsv[..., 1].ravel() / 255]


def collect(gids, max_frames=40):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import validate as V
    X, y, grp, meta = [], [], [], []
    for gid in gids:
        p = V.load(gid)
        if p is None:
            continue
        p = p[(p.clock >= 120) & (p['loc'] >= 0.45)]
        for _, r in p.head(max_frames).iterrows():
            im = cv2.imread(f'{D}/v3/crops/{gid}/{r.sec}.png')
            cm = mm.canon(im, mm.locate(im))
            for i in range(1, 11):
                if not np.isfinite(r[f'x{i}']):
                    continue
                x, yv = r[f'x{i}'] * mm.N, r[f'y{i}'] * mm.N
                X.append(np.r_[r[f's{i}'], r[f'c{i}'], r[f'm{i}'], patch_feats(cm, x, yv, r.icon_d)])
                y.append(int(r[f'p{i}_currentHealth'] > 0))
                grp.append(gid)
    return np.array(X), np.array(y), np.array(grp)


def main():
    gids = [os.path.basename(f)[:-8] for f in glob.glob(f'{D}/v3/parsed/*.parquet')]
    X, y, g = collect(gids)
    print('candidates', len(y), 'alive share', y.mean().round(3), 'games', len(set(g)))
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    print('AUC alive vs dead: template score only', round(roc_auc_score(y, X[:, 0]), 3), ' classifier', round(roc_auc_score(y, oof), 3))
    na, nd = y.sum(), (1 - y).sum()
    for t in (0.8, 0.9, 0.95, 0.97, 0.98):
        k = oof >= t
        print(f'  p>={t}: keeps {(k & (y == 1)).sum() / na:.3f} of alive candidates, {(k & (y == 0)).sum() / nd:.3f} of dead ones')
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0).fit(X, y)
    pickle.dump(m, open(f'{D}/v3/ref/icon_clf.pkl', 'wb'))


if __name__ == '__main__':
    main()
