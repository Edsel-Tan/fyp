"""v3 step 5d: train the minimap icon detector on synthetic maps (synth.py); CenterNet-style heatmaps, 2 classes.

Net: small U-Net (3 down / 3 up, 32-128 channels) on the 256 x 256 canonical map, output stride 1, sigmoid heatmaps
for the blue and red icon centres; focal loss (CenterNet, alpha 2, beta 4), Gaussian targets with sigma = d / 6.
Backgrounds: median backgrounds of the extracted games (data/v3/bgs.npy, by make_bgs()); games are split so
validation backgrounds are unseen. Saves data/v3/ref/det.pt.
usage (from esports/src): python v3/train_det.py [STEPS]
"""
import os, sys, glob, time
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth, minimap as mm

D = '../data'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_bgs(max_games=400):
    import pandas as pd
    out = []
    for g in sorted(glob.glob(f'{D}/v3/crops/*/done.json'))[:max_games]:
        d = os.path.dirname(g)
        fs = sorted(glob.glob(f'{d}/[0-9]*.png'))[::3]
        if len(fs) < 6:
            continue
        ims = [cv2.imread(x) for x in fs]
        l = mm.locate(ims[len(ims) // 2])
        if l[0] < 0.45:
            continue
        C = []
        for im in ims:
            li = mm.locate(im, l[1:])
            if li[0] >= 0.45:
                C.append(mm.canon(im, li))
        if len(C) >= 5:
            out.append(np.median(np.stack(C), 0).astype(np.uint8))
    np.save(f'{D}/v3/bgs.npy', np.stack(out))
    return np.stack(out)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = block(3, 32), block(32, 64), block(64, 96), block(96, 128)
        self.d3, self.d2, self.d1 = block(128 + 96, 96), block(96 + 64, 64), block(64 + 32, 32)
        self.out = nn.Conv2d(32, 2, 1)
        self.out.bias.data.fill_(-4.0)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        u = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1)
        d3 = self.d3(u(e4, e3))
        d2 = self.d2(u(d3, e2))
        d1 = self.d1(u(d2, e1))
        return self.out(d1)


def to_t(img):
    return torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1).float() / 255


def target(labels, d):
    H = np.zeros((2, mm.N, mm.N), np.float32)
    yy, xx = np.mgrid[:mm.N, :mm.N]
    s = d / 6
    for x, y, t in labels:
        H[t] = np.maximum(H[t], np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)))
    return H


def focal(logit, gt):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float() if False else (gt > 0.99).float()
    neg = 1 - pos
    lp = -torch.log(p) * (1 - p) ** 2 * pos
    ln = -torch.log(1 - p) * p ** 2 * (1 - gt) ** 4 * neg
    return (lp.sum() + ln.sum()) / pos.sum().clamp(min=1)


def peaks(hm, thr=0.3, k=7):
    """hm (2, H, W) numpy probabilities -> list of (x, y, team, p)."""
    out = []
    for t in (0, 1):
        h = hm[t]
        mx = cv2.dilate(h, np.ones((k, k), np.uint8))
        ys, xs = np.nonzero((h >= mx) & (h >= thr))
        out += [(float(x), float(y), t, float(h[y, x])) for x, y in zip(xs, ys)]
    return out


def batch(S, n):
    X, Y = [], []
    for _ in range(n):
        img, lab, d = S.sample()
        X.append(to_t(img))
        Y.append(torch.from_numpy(target(lab, d)))
    return torch.stack(X), torch.stack(Y)


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    bgs = np.load(f'{D}/v3/bgs.npy') if os.path.exists(f'{D}/v3/bgs.npy') else make_bgs()
    nv = max(2, len(bgs) // 6)
    Str, Sva = synth.Synth(bgs[nv:], 0), synth.Synth(bgs[:nv], 1)
    print(f'backgrounds {len(bgs)} (val {nv}), device {dev}', flush=True)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(1)
    nxt = ex.submit(batch, Str, 16)
    t0 = time.time()
    for it in range(steps):
        X, Y = nxt.result()
        nxt = ex.submit(batch, Str, 16)
        net.train()
        loss = focal(net(X.to(dev)), Y.to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0 or it == steps - 1:
            net.eval()
            tp = fp = fn = 0
            with torch.no_grad():
                for _ in range(8):
                    img, lab, d = Sva.sample()
                    hm = torch.sigmoid(net(to_t(img)[None].to(dev)))[0].cpu().numpy()
                    pk = peaks(hm, 0.4)
                    used = set()
                    for x, y, t in lab:
                        m = [k for k, (px, py, pt, _) in enumerate(pk) if pt == t and k not in used and np.hypot(px - x, py - y) < d / 3]
                        if m:
                            used.add(m[0]); tp += 1
                        else:
                            fn += 1
                    fp += len(pk) - len(used)
            print(f'{it} loss {loss.item():.3f} val recall {tp / max(tp + fn, 1):.3f} precision {tp / max(tp + fp, 1):.3f} '
                  f'{(time.time() - t0) / 60:.1f} min', flush=True)
    torch.save(net.state_dict(), f'{D}/v3/ref/det.pt')


if __name__ == '__main__':
    main()
