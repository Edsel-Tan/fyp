#!/usr/bin/env python3
"""The one pre-registered test in pm/PREREG_drift.md. Read that file first.

Reuses pm/pmv1_text_signal.py's own pieces -- load, price_block, zscore, stack,
best_ridge, ic, book_rows, apply_null -- so the only thing that differs from the
sec.3.6 drift arm is the split, which the pre-registration fixes:

  train   archive rows before 2026-02-27, events first seen before then
  valid   archive rows in [2026-02-27, 2026-04-29), events first seen in that window
  test    data/prereg/panel.parquet (pm/pmlive_drift_panel.py): 2026-05-01 onward,
          events the archive never saw

    python pm/prereg_drift_test.py            # the test
    python pm/prereg_drift_test.py --null     # the secondary shuffle null
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pmv1_text_signal as S
from pmv1_text_embed import embed, MODELS

ROOT = os.path.join(HERE, os.pardir)
PRE = os.path.join(ROOT, "data", "prereg")
V0, V1, T0 = pd.Timestamp("2026-02-27"), pd.Timestamp("2026-04-29"), pd.Timestamp("2026-05-01")
N_BOOT = 10_000


def new_rows(cats):
    d = pd.read_parquet(os.path.join(PRE, "panel.parquet"))
    d = d.dropna(subset=["y_fwd"]).copy()
    lc = {c.lower(): c for c in cats}
    d["cat"] = d.tags.map(lambda s: next((lc[t.lower()] for t in json.loads(s or "[]")
                                          if t and t.lower() in lc), "Other"))
    d["date"] = pd.to_datetime(d.ts, unit="s")
    d = d[d.date >= T0].copy()
    d["y"] = d.y_fwd
    d["w"] = 1.0 / d.groupby("market_id").market_id.transform("size")
    d["ev_first"] = d.groupby("event_id").date.transform("min")
    return d


def boot_dic(p1, p2, y, ev, n=N_BOOT, seed=0):
    """Event-clustered bootstrap of IC(p2) - IC(p1), from per-event sufficient stats."""
    e_codes, e = np.unique(ev, return_inverse=True)
    E = len(e_codes)
    def sums(x):
        return np.stack([np.bincount(e, w, minlength=E) for w in
                         (np.ones_like(y), x, y, x * x, y * y, x * y)], 1)
    A, B = sums(p1), sums(p2)
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(0, n, 500):
        m = min(500, n - k)
        W = rng.multinomial(E, np.full(E, 1 / E), size=m).astype(float)   # [m, E]
        def corr(Z):
            s = W @ Z
            N, sx, sy, sxx, syy, sxy = s.T
            cov = sxy / N - sx * sy / N ** 2
            return cov / np.sqrt((sxx / N - (sx / N) ** 2) * (syy / N - (sy / N) ** 2))
        out[k:k + m] = corr(B) - corr(A)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", action="store_true", help="secondary: shuffle-null the text")
    a = ap.parse_args()

    arc = S.load("nl", "fwd")
    arc = arc[arc.date < V1].copy()
    cats = arc.cat.value_counts().head(20).index.tolist()
    new = new_rows(cats)
    assert not set(new.event_id) & set(arc.event_id), "test events leaked into the archive"
    arc["is_new"], new["is_new"] = False, True
    df = pd.concat([arc, new[arc.columns.intersection(new.columns).tolist() + ["tags"]]],
                   ignore_index=True)
    df = S.apply_null(df, "shuffle" if a.null else "none", 0)
    print(f"archive {(~df.is_new).sum():,} rows / new {df.is_new.sum():,} rows, "
          f"{df[df.is_new].market_id.nunique():,} markets, {df[df.is_new].event_id.nunique():,} events")

    tr = ((df.ev_first < V0) & (df.date < V0) & ~df.is_new).values
    va = ((df.ev_first >= V0) & (df.ev_first < V1) & (df.date >= V0) & (df.date < V1) & ~df.is_new).values
    te = df.is_new.values

    # FinBERT block: archive rows from the stored embeddings, new questions embedded
    # by the same function (mean-pooled, max length 64)
    z = np.load(os.path.join(S.NLP, "emb_finbert_text.npz"))
    lut = dict(zip(z["market_id"], range(len(z["market_id"]))))
    txt = df.drop_duplicates("market_id_txt").set_index("market_id_txt").text
    missing = [m for m in txt.index if m not in lut]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E_new = embed(txt.loc[missing].tolist(), MODELS["finbert"], dev) if missing else np.zeros((0, z["emb"].shape[1]))
    emb = np.vstack([z["emb"], E_new])
    lut.update({m: len(z["market_id"]) + i for i, m in enumerate(missing)})
    Xt = emb[df.market_id_txt.map(lut).values].astype(np.float64)

    Xp = S.zscore(S.price_block(df, cats), tr)
    Xt = S.zscore(Xt, tr)
    y, w, ev = df.y.values, df.w.values, df.event_id.values
    v1, al1, _, m1 = S.best_ridge(lambda g: Xp, y, w, tr, va, ev[va], [0.0])
    v2, al2, g2, m2 = S.best_ridge(lambda g: S.stack(Xp, Xt, g), y, w, tr, va, ev[va], S.GAMMAS)
    p1 = m1.predict(Xp[te]); p2 = m2.predict(S.stack(Xp, Xt, g2)[te])
    r1, r2 = S.ic(p1, y[te], ev[te]), S.ic(p2, y[te], ev[te])
    dic = r2["ic"] - r1["ic"]
    bs = boot_dic(p1, p2, y[te], ev[te]) if g2 > 0 else np.zeros(N_BOOT)
    p = float((bs <= 0).mean())
    supported = bool(dic > 0 and p < 0.05 and g2 > 0)

    d_te = df[te].reset_index(drop=True)
    month = d_te.date.dt.to_period("M").astype(str).values
    by_m = {}
    for mo in sorted(set(month)):
        k = month == mo
        if k.sum() > 200:
            by_m[mo] = dict(n=int(k.sum()), events=int(pd.unique(ev[te][k]).size),
                            dic=S.ic(p2[k], y[te][k], ev[te][k])["ic"] - S.ic(p1[k], y[te][k], ev[te][k])["ic"])
    books = {k: S.book_stats([S.book_rows(pp, d_te)]) for k, pp in (("M1", p1), ("M2", p2))}

    res = dict(prereg_sha256="d35a86df6a0e1983d27f6a0a707e0156f04640ce5ec823f09912598a32876ee0",
               null=a.null, alpha1=al1, alpha2=al2, gamma2=g2, valid_ic1=v1, valid_ic2=v2,
               n_train=int(tr.sum()), n_valid=int(va.sum()), n_test=int(te.sum()),
               test_events=int(pd.unique(ev[te]).size),
               ic1=r1["ic"], ic2=r2["ic"], dic=dic, boot_p=p,
               boot_ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
               supported=supported, by_month=by_m, books=books)
    print(json.dumps(res, indent=1, default=float))
    out = os.path.join(PRE, "result_null.json" if a.null else "result.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    print("wrote", out)


if __name__ == "__main__":
    main()
