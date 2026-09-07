#!/usr/bin/env python3
"""Block-aware forward-return forecaster for the crypto panel.

Same model as hrt/forecast.py -- encoder-only Transformer over a 10-bar lookback
of the 158 Alpha158 features, linear head, MSE on the cross-sectionally
standardised label -- but every lookback window and every label is confined to a
single contiguous block, because the findata crypto tape is cut by month-long
holes and a window that straddles one is fiction.

  --label causal  features at bar t  -> open_{t+1} -> open_{t+2}   (tradeable)
  --label paper   the HRT paper's literal timing, open_t -> open_{t+1}
"""
import argparse, os, sys, time
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "hrt"))
from forecast import Encoder, robust_zscore, cs_zscore, gather_windows, daily_ic   # noqa: E402

ART = os.path.join(HERE, "artifacts")
TRAIN_END, VALID_END = "2024-12-31", "2025-12-31"


def build_labels(op, block, mode):
    T = op.shape[0]
    lab = np.full_like(op, np.nan, dtype=np.float64)
    b = np.asarray(block)
    if mode == "causal":
        ok = np.zeros(T, bool); ok[:T - 2] = (b[:T - 2] == b[2:])
        lab[:T - 2] = np.where(ok[:T - 2, None], op[2:] / op[1:T - 1] - 1.0, np.nan)
    else:
        ok = np.zeros(T, bool); ok[:T - 1] = (b[:T - 1] == b[1:])
        lab[:T - 1] = np.where(ok[:T - 1, None], op[1:] / op[:T - 1] - 1.0, np.nan)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(ART, "panel_1hour.npz"))
    ap.add_argument("--label", choices=["causal", "paper"], default="causal")
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ART, f"fr_{a.label}.npz")

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.panel, allow_pickle=True)
    X = z["X"]; dates = z["dates"].astype(str); block = z["block"]
    op = z["open"].astype(np.float64); syms = z["symbols"].astype(str)
    T, N, F = X.shape
    print(f"panel {X.shape}  {dates[0]} .. {dates[-1]}  blocks={len(np.unique(block))}  dev={dev}")

    tr = np.where(dates <= TRAIN_END)[0]
    va = np.where((dates > TRAIN_END) & (dates <= VALID_END))[0]
    te = np.where(dates > VALID_END)[0]
    print(f"train {len(tr)}  valid {len(va)}  test {len(te)} bars")

    Xn, _, _ = robust_zscore(X.astype(np.float64), tr)
    lab_raw = build_labels(op, block, a.label)
    lab = cs_zscore(lab_raw)
    valid_mask = np.isfinite(lab)

    Xg = torch.tensor(Xn, dtype=torch.float32, device=dev)
    Lg = torch.tensor(np.nan_to_num(lab), dtype=torch.float32, device=dev)

    def pairs(day_idx):
        # a lookback window may not cross a block boundary
        lo = a.lookback - 1
        ok = [t for t in day_idx if t >= lo and block[t - lo] == block[t]]
        ts, ns = np.meshgrid(np.array(ok), np.arange(N), indexing="ij")
        ts, ns = ts.ravel(), ns.ravel()
        keep = valid_mask[ts, ns]
        return ts[keep], ns[keep]

    tr_t, tr_n = pairs(tr); va_t, va_n = pairs(va)
    print(f"samples: train {len(tr_t):,}  valid {len(va_t):,}")

    model = Encoder(F, lookback=a.lookback).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.MSELoss()
    g = lambda x: torch.tensor(x, device=dev)
    tr_t_g, tr_n_g, va_t_g, va_n_g = g(tr_t), g(tr_n), g(va_t), g(va_n)

    def predict(t_g, n_g):
        model.eval(); o = []
        with torch.no_grad():
            for i in range(0, len(t_g), 32768):
                o.append(model(gather_windows(Xg, t_g[i:i+32768], n_g[i:i+32768],
                                              a.lookback)).float().cpu().numpy())
        return np.concatenate(o) if o else np.array([])

    best, best_state, bad = -9e9, None, 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr_t_g), device=dev)
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(perm), a.batch):
            idx = perm[i:i + a.batch]
            loss = lossf(model(gather_windows(Xg, tr_t_g[idx], tr_n_g[idx], a.lookback)),
                         Lg[tr_t_g[idx], tr_n_g[idx]])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
            tot += loss.item(); nb += 1
        ic, icstd = daily_ic(predict(va_t_g, va_n_g), lab[va_t, va_n], va_t)
        flag = ""
        if ic > best:
            best, best_state, bad, flag = ic, {k: v.detach().clone() for k, v in
                                               model.state_dict().items()}, 0, " *"
        else:
            bad += 1
        print(f"ep {ep:02d} loss {tot/nb:.4f} valIC {ic:+.4f} {time.time()-t0:.0f}s{flag}")
        if bad >= 5:
            print("early stop"); break
    model.load_state_dict(best_state)

    lo = a.lookback - 1
    ok_t = np.array([t for t in range(lo, T) if block[t - lo] == block[t]])
    all_t, all_n = np.meshgrid(ok_t, np.arange(N), indexing="ij")
    all_t, all_n = all_t.ravel(), all_n.ravel()
    p = predict(g(all_t), g(all_n))
    fr = np.full((T, N), np.nan, np.float32)
    fr[all_t, all_n] = p

    res = {}
    for name, idx in (("train", tr), ("valid", va), ("test", te)):
        t, n = pairs(idx)
        ic, s = daily_ic(fr[t, n], lab[t, n], t)
        res[name] = ic
        print(f"{name:6s} IC {ic:+.4f}  ICIR {ic/max(s,1e-9):+.3f}  bars {len(np.unique(t))}")

    np.savez_compressed(out, fr=fr, dates=z["dates"], symbols=syms, block=block,
                        label=lab_raw.astype(np.float32), mode=a.label,
                        ic_train=res["train"], ic_valid=res["valid"], ic_test=res["test"])
    print("wrote", out)


if __name__ == "__main__":
    main()
