#!/usr/bin/env python3
"""Transformer-encoder forward-return predictor -- the HLC's signal source.

Per the paper: encoder-only Transformer over a 10-day lookback of the 158 Qlib
Alpha158 features, linear head, supervised on the training split.

Timing convention (`--label causal`, default): features observed at the close of
day t predict the open-to-open return from day t+1 to day t+2, which is the
return an agent trading at the open of day t+1 can actually capture.
`--label paper` reproduces the paper's literal wording (open_t -> open_{t+1}),
which is already realised by the time the trade is placed.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

TRAIN = ("2015-01-01", "2019-12-31")
VALID = ("2020-01-01", "2020-12-31")
TEST = ("2021-01-01", "2022-12-31")


def split_idx(dates, lo, hi):
    return np.where((dates >= lo) & (dates <= hi))[0]


class Encoder(nn.Module):
    def __init__(self, n_feat, d_model=64, nhead=4, layers=2, dropout=0.1, lookback=10):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                      # x [B, L, F]
        z = self.enc(self.proj(x) + self.pos)
        return self.head(z[:, -1]).squeeze(-1)


def build_labels(op, mode):
    """op [T,N] open prices -> label [T,N] keyed by the feature day t."""
    T = op.shape[0]
    lab = np.full_like(op, np.nan, dtype=np.float64)
    if mode == "causal":                       # open_{t+1} -> open_{t+2}
        lab[:T - 2] = op[2:] / op[1:T - 1] - 1.0
    else:                                      # paper: open_t -> open_{t+1}
        lab[:T - 1] = op[1:] / op[:T - 1] - 1.0
    return lab


def robust_zscore(X, fit_idx, clip=3.0):
    """Qlib RobustZScoreNorm: (x - median) / (1.4826 * MAD), clipped."""
    ref = X[fit_idx].reshape(-1, X.shape[-1])
    med = np.nanmedian(ref, 0)
    mad = np.nanmedian(np.abs(ref - med), 0) * 1.4826
    mad = np.where(mad < 1e-8, 1.0, mad)
    out = (X - med) / mad
    np.clip(out, -clip, clip, out=out)
    return np.nan_to_num(out, nan=0.0), med, mad


def cs_zscore(lab):
    """Cross-sectional z-score of the label within each date."""
    m = np.nanmean(lab, 1, keepdims=True)
    s = np.nanstd(lab, 1, keepdims=True)
    return (lab - m) / np.where(s < 1e-12, 1.0, s)


def gather_windows(Xg, t_idx, n_idx, lookback):
    """Xg [T,N,F] on device -> [B, lookback, F] ending at t_idx (inclusive)."""
    offs = torch.arange(-lookback + 1, 1, device=Xg.device)
    tt = t_idx[:, None] + offs[None, :]                  # [B, L]
    nn_ = n_idx[:, None].expand(-1, lookback)
    return Xg[tt, nn_]


def daily_ic(pred, true, t_idx):
    """Mean cross-sectional Spearman-free (Pearson) IC by date."""
    out = []
    for t in np.unique(t_idx):
        m = t_idx == t
        p, y = pred[m], true[m]
        if len(p) > 5 and p.std() > 1e-12 and y.std() > 1e-12:
            out.append(np.corrcoef(p, y)[0, 1])
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(ART, "panel.npz"))
    ap.add_argument("--label", choices=["causal", "paper"], default="causal")
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ART, f"fr_{a.label}.npz")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.panel, allow_pickle=True)
    X, dates, syms, op = z["X"], z["dates"].astype(str), z["symbols"].astype(str), z["open"].astype(np.float64)
    T, N, F = X.shape
    print(f"panel {X.shape}  {dates[0]}..{dates[-1]}  device={dev}")

    tr = split_idx(dates, *TRAIN); va = split_idx(dates, *VALID); te = split_idx(dates, *TEST)
    print(f"train {len(tr)}  valid {len(va)}  test {len(te)} days")

    Xn, med, mad = robust_zscore(X.astype(np.float64), tr)
    lab_raw = build_labels(op, a.label)
    lab = cs_zscore(lab_raw)

    Xg = torch.tensor(Xn, dtype=torch.float32, device=dev)
    Lg = torch.tensor(np.nan_to_num(lab), dtype=torch.float32, device=dev)
    valid_mask = np.isfinite(lab)

    def pairs(day_idx):
        lo = a.lookback - 1
        ok = [t for t in day_idx if t >= lo]
        ts, ns = np.meshgrid(np.array(ok), np.arange(N), indexing="ij")
        ts, ns = ts.ravel(), ns.ravel()
        keep = valid_mask[ts, ns]
        return ts[keep], ns[keep]

    tr_t, tr_n = pairs(tr); va_t, va_n = pairs(va)
    print(f"samples: train {len(tr_t):,}  valid {len(va_t):,}")

    model = Encoder(F, lookback=a.lookback).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.MSELoss()
    tr_t_g = torch.tensor(tr_t, device=dev); tr_n_g = torch.tensor(tr_n, device=dev)
    va_t_g = torch.tensor(va_t, device=dev); va_n_g = torch.tensor(va_n, device=dev)

    def predict(t_g, n_g):
        model.eval(); outp = []
        with torch.no_grad():
            for i in range(0, len(t_g), 32768):
                xb = gather_windows(Xg, t_g[i:i + 32768], n_g[i:i + 32768], a.lookback)
                outp.append(model(xb).float().cpu().numpy())
        return np.concatenate(outp)

    best, best_state, bad = -9e9, None, 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr_t_g), device=dev)
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            xb = gather_windows(Xg, tr_t_g[idx], tr_n_g[idx], a.lookback)
            yb = Lg[tr_t_g[idx], tr_n_g[idx]]
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            tot += loss.item(); nb += 1
        ic, icstd = daily_ic(predict(va_t_g, va_n_g), lab[va_t, va_n], va_t)
        flag = ""
        if ic > best:
            best, best_state, bad, flag = ic, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0, " *"
        else:
            bad += 1
        print(f"ep {ep:02d} loss {tot/nb:.4f} valIC {ic:+.4f} (ICIR {ic/max(icstd,1e-9):+.3f}) "
              f"{time.time()-t0:.0f}s{flag}")
        if bad >= 6:
            print("early stop"); break

    model.load_state_dict(best_state)

    # score every (day, stock) so the RL env can index freely
    lo = a.lookback - 1
    all_t, all_n = np.meshgrid(np.arange(lo, T), np.arange(N), indexing="ij")
    all_t, all_n = all_t.ravel(), all_n.ravel()
    p = predict(torch.tensor(all_t, device=dev), torch.tensor(all_n, device=dev))
    fr = np.full((T, N), np.nan, np.float32)
    fr[all_t, all_n] = p

    for name, idx in (("train", tr), ("valid", va), ("test", te)):
        t, n = pairs(idx)
        ic, s = daily_ic(fr[t, n], lab[t, n], t)
        print(f"{name:6s} IC {ic:+.4f}  ICIR {ic/max(s,1e-9):+.3f}  n_days {len(np.unique(t))}")

    np.savez_compressed(out, fr=fr, dates=dates, symbols=syms,
                        label=lab_raw.astype(np.float32), mode=a.label, val_ic=best)
    print("wrote", out)


if __name__ == "__main__":
    main()
