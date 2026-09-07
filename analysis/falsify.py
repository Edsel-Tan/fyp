#!/usr/bin/env python3
"""Falsification audit of the HRT pipeline (PAPERS.md [10]).

The audit runs the *complete* workflow -- Alpha158 features, Transformer
forecaster, top-k long book -- against synthetic panels in which returns are a
martingale difference, so there is provably nothing to predict. A workflow that
still reports out-of-sample evidence on that reference class is falsified.

Reported for each arm:
  * test IC on the synthetic panels (the null distribution)
  * test IC on the real panel (the claim being audited)
  * the top-k strategy return the pipeline reports under each
  * the magnitude gap between optimised in-sample and disjoint out-of-sample
    evidence, which [10] uses to quantify selection-induced inflation
"""
import glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "hrt"))
from forecast import daily_ic, cs_zscore                     # noqa: E402
from env import metrics                                      # noqa: E402

ART = os.path.join(HERE, os.pardir, "hrt", "artifacts")
SPLITS = {"train": ("2015-01-01", "2019-12-31"), "valid": ("2020-01-01", "2020-12-31"),
          "test": ("2021-01-01", "2022-12-31")}


def ic_by_split(frfile):
    z = np.load(frfile, allow_pickle=True)
    fr, lab, dates = z["fr"], cs_zscore(z["label"].astype(np.float64)), z["dates"].astype(str)
    out = {}
    for k, (lo, hi) in SPLITS.items():
        idx = np.where((dates >= lo) & (dates <= hi))[0]
        t, n = np.meshgrid(idx, np.arange(fr.shape[1]), indexing="ij")
        t, n = t.ravel(), n.ravel()
        m = np.isfinite(fr[t, n]) & np.isfinite(lab[t, n])
        out[k] = daily_ic(fr[t[m], n[m]], lab[t[m], n[m]], t[m])[0]
    return out


def topk_long(prices, fr, k=30, cost=0.001):
    T, N = prices.shape
    w, vals = np.zeros(N), [1.0]
    for t in range(T - 1):
        w_new = np.zeros(N)
        w_new[np.argsort(-fr[t])[:k]] = 1.0 / k
        turn = np.abs(w_new - w).sum()
        r = float(w_new @ (prices[t + 1] / prices[t] - 1.0))
        vals.append(vals[-1] * (1 + r - cost * turn))
        w = w_new * (prices[t + 1] / prices[t])
        s = w.sum()
        w = w / s if s > 0 else w
    return np.array(vals)


def strat(panel, frfile, k=30):
    zp = np.load(panel, allow_pickle=True)
    zf = np.load(frfile, allow_pickle=True)
    dates = zp["dates"].astype(str)
    op = zp["open"].astype(np.float64)
    fr = np.nan_to_num(zf["fr"].astype(np.float64))
    fr = np.vstack([np.zeros((1, fr.shape[1])), fr[:-1]])      # tradeable at t+1's open
    out = {}
    for yr in ("2021", "2022"):
        i = np.where((dates >= f"{yr}-01-01") & (dates <= f"{yr}-12-31"))[0]
        out[yr] = metrics(topk_long(op[i], fr[i], k))
    return out


def main():
    real = {m: (os.path.join(ART, "panel.npz"), os.path.join(ART, f"fr_{m}.npz"))
            for m in ("causal", "paper")}
    synth = {}
    for f in sorted(glob.glob(os.path.join(ART, "fr_synth_*.npz"))):
        b = os.path.basename(f)[len("fr_synth_"):-4]
        mode, seed = (b.split("_s") + ["0"])[:2]
        p = os.path.join(ART, "panel_synth.npz" if seed == "0"
                         else f"panel_synth_s{seed}.npz")
        if os.path.exists(p):
            synth.setdefault(mode, []).append((p, f))

    print(f"{'panel':<22}{'label':<8}{'IC train':>10}{'IC valid':>10}{'IC test':>10}"
          f"{'2021 cum':>10}{'2021 Sh':>9}{'2022 cum':>10}{'2022 Sh':>9}")
    print("-" * 98)
    rows = {}
    for mode, (p, f) in real.items():
        if not os.path.exists(f):
            continue
        ic, st = ic_by_split(f), strat(p, f)
        rows[("real", mode)] = (ic, st)
        print(f"{'REAL S&P 500':<22}{mode:<8}{ic['train']:>+10.4f}{ic['valid']:>+10.4f}"
              f"{ic['test']:>+10.4f}{st['2021']['cum_return']:>+10.3f}"
              f"{st['2021']['sharpe']:>+9.2f}{st['2022']['cum_return']:>+10.3f}"
              f"{st['2022']['sharpe']:>+9.2f}")
    null = {}
    for mode, items in sorted(synth.items()):
        acc = []
        for i, (p, f) in enumerate(items):
            ic, st = ic_by_split(f), strat(p, f)
            acc.append((ic, st))
            print(f"{'synthetic (null) #' + str(i):<22}{mode:<8}{ic['train']:>+10.4f}"
                  f"{ic['valid']:>+10.4f}{ic['test']:>+10.4f}"
                  f"{st['2021']['cum_return']:>+10.3f}{st['2021']['sharpe']:>+9.2f}"
                  f"{st['2022']['cum_return']:>+10.3f}{st['2022']['sharpe']:>+9.2f}")
        null[mode] = acc

    print("\n=== verdict ===")
    for mode in sorted(null):
        te = np.array([a[0]["test"] for a in null[mode]])
        s21 = np.array([a[1]["2021"]["cum_return"] for a in null[mode]])
        print(f"\n[{mode} timing]  null test IC over {len(te)} synthetic panels: "
              f"mean {te.mean():+.4f}  sd {te.std(ddof=1) if len(te) > 1 else float('nan'):.4f}"
              f"  range [{te.min():+.4f}, {te.max():+.4f}]")
        print(f"                null 2021 top-30 return: mean {s21.mean():+.2%}  "
              f"range [{s21.min():+.2%}, {s21.max():+.2%}]")
        if ("real", mode) in rows:
            r = rows[("real", mode)][0]["test"]
            if len(te) > 1 and te.std(ddof=1) > 0:
                z = (r - te.mean()) / te.std(ddof=1)
                print(f"                real test IC {r:+.4f}  ->  z vs null = {z:+.2f}")
            print("                VERDICT: pipeline FALSIFIED on this reference class"
                  if abs(te.mean()) > 0.05 else
                  "                VERDICT: pipeline survives -- reports ~zero where there is zero")
    json.dump({f"{k[0]}_{k[1]}": v[0] for k, v in rows.items()},
              open(os.path.join(ART, "falsify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
