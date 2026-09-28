#!/usr/bin/env python3
"""Does the *question text* predict a prediction market, beyond its price? ([16])

The setting is the reason to ask. Every NLP-in-finance result on the reading list
reads text *about* an asset -- news, filings, tweets -- and maps sentiment onto a
ticker that exists independently of the words. A prediction market has no such
underlying: the question *is* the contract, and two markets differ only in their
sentences. So the honest form of "can FinBERT predict price direction here" is a
nested one, and it is the only form this script reports:

    M1  price state + category           the baseline that must be beaten
    M2  M1 + text features               the same thing plus the sentence
    dIC = IC(M2) - IC(M1)                what the language is worth

Reporting IC(M2) alone would be meaningless. RESULTS.md sec.3 already shows the
price path carries a large, strongly miscalibrated signal -- favourites at
p in [0.6,0.9] are overpriced by ~4 pp -- and any encoder that can tell "will
trump win the 2024 us presidential election" from "nba bos bkn total 2pt5"
recovers a chunk of that for free, because category and typical price level are
written into the words. That is not a text alpha.

Two design points make the nested comparison fair rather than rigged:

  Block-wise shrinkage. A single ridge penalty over 31 price columns and 768
  embedding columns cannot shrink the text without also shrinking the price, so
  M2 would lose to M1 as a fitting artifact rather than as a finding. The text
  block is therefore scaled by gamma before stacking -- equivalent to penalising
  it at alpha/gamma^2 -- and (alpha, gamma) are searched jointly. gamma = 0
  reproduces M1 exactly, so M2 can only lose on validation by being genuinely
  worse, and a negative *test* dIC is then honest overfitting, which is itself
  the result worth reporting.

  Walk-forward folds. A single train/valid/test cut on this panel puts ~1.2k
  events in validation against ~5.5k in test, and the archive's market count
  grows two orders of magnitude across it, so one cut measures one regime. Each
  fold trains on everything before a validation window and tests on the month
  after it; events are assigned on first appearance and rows must fall inside
  their own event's window, so no event straddles a boundary in either direction.

Three falsification arms, in the spirit of sec.4:

  --null shuffle    permute the question across markets within (fold x category x
                    price decile). Language is destroyed, everything else held.
                    dIC must collapse. If it does not, the pipeline is
                    manufacturing the result and nothing else it reports counts.
  --split random    replace the walk-forward split with an i.i.d. *row* split --
                    the split an off-the-shelf pipeline would use -- and with
                    `eventrandom`, an i.i.d. *event* split. The three together
                    decompose the leak into the part that is the same market on
                    both sides and the part that is merely contemporaneous.
  --kind ts|date    restrict to the machine-templated strata, whose slugs carry a
                    unix timestamp or an ISO date. There the "text" is a clock.

Economics, in the pp-per-share units of sec.3.5. The terminal target is bought
and held to resolution, and settlement pays at par, so the trip crosses the
spread *once* rather than twice -- structurally cheaper than sec.3.5's two-legged
round trip, which lost 4.2x its gross edge to the exit. Execution is at the side
a taker must actually hit:

    net_pp = side * (win_event - exec),  exec = vwap_buy if side=+1 else vwap_sell

Significance is always clustered on event_id. The panel has 460k rows but ~9.9k
events, the 2024 presidential complex alone carries tens of thousands of rows,
and every row of a market shares one resolution -- so row-level t-statistics here
overstate by roughly sqrt(rows/events) ~ 7.

    python pm/pmv1_text_signal.py --text tfidf
    python pm/pmv1_text_signal.py --text finbert --residual
    python pm/pmv1_text_signal.py --text finbert --null shuffle
"""
import argparse, json, os, sys
import duckdb, numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
NLP = os.path.join(ROOT, "data", "nlp")

ALPHAS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5]
GAMMAS = [0.0, 0.05, 0.2, 1.0]          # 0 reproduces M1 exactly
VALID_DAYS = 61                          # validation window ahead of each test month


# ------------------------------------------------------------------ utilities
def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def event_mean(v, event_id, w=None):
    """One number per event: the weighted mean of its rows."""
    df = pd.DataFrame({"v": np.asarray(v, float), "e": np.asarray(event_id),
                       "w": 1.0 if w is None else np.asarray(w, float)})
    s = df.groupby("e").apply(lambda g: np.average(g.v, weights=g.w), include_groups=False)
    return s


def event_stats(v, event_id, w=None):
    g = event_mean(v, event_id, w)
    n = len(g)
    if n < 3:
        return dict(mean_pp=float("nan"), t=float("nan"), n_events=int(n))
    return dict(mean_pp=100 * float(g.mean()),
                t=float(g.mean() / (g.std(ddof=1) / np.sqrt(n))), n_events=int(n))


def ic(pred, y, event_id):
    """Pearson IC, with an event-clustered t on its per-event contribution."""
    pred = np.asarray(pred, float)
    if np.std(pred) < 1e-12 or len(pred) < 10:
        return dict(ic=0.0, ic_t=0.0)
    r = float(np.corrcoef(pred, y)[0, 1])
    a = (pred - pred.mean()) / (pred.std() + 1e-12)
    b = (y - y.mean()) / (y.std() + 1e-12)
    return dict(ic=r, ic_t=event_stats(a * b, event_id)["t"])


# --------------------------------------------------------------------- panel
def load(kind, target):
    df = duckdb.sql(f"SELECT * FROM read_parquet('{NLP}/panel.parquet')").df()
    if kind != "all":
        df = df[df.kind == kind]
    col = "y_term" if target == "term" else "y_fwd"
    df = df.dropna(subset=[col]).copy()
    df["date"] = pd.to_datetime(df.ts, unit="s")
    df["y"] = df[col]
    # each market carries total weight 1, so a 470-day mega-market does not
    # outvote 470 one-day markets during fitting
    df["w"] = 1.0 / df.groupby("market_id").market_id.transform("size")
    df["ev_first"] = df.groupby("event_id").date.transform("min")
    return df.reset_index(drop=True)


def make_folds(df, mode, n_folds, seed):
    """Yield (name, train_mask, valid_mask, test_mask)."""
    if mode in ("random", "eventrandom"):
        rng = np.random.default_rng(seed)
        if mode == "random":
            u = rng.random(len(df))
            lab = pd.Series(np.where(u < .6, "train", np.where(u < .75, "valid", "test")),
                            index=df.index)
        else:
            ev = df.event_id.unique()
            m = pd.Series(rng.choice(["train", "valid", "test"], len(ev), p=[.6, .15, .25]),
                          index=ev)
            lab = df.event_id.map(m)
        yield ("iid", (lab == "train").values, (lab == "valid").values,
               (lab == "test").values)
        return

    # walk-forward: monthly test windows over the tail of the panel
    end = df.date.max()
    edges = pd.date_range(end=end.normalize() + pd.Timedelta(days=1),
                          periods=n_folds + 1, freq="MS")
    for i in range(len(edges) - 1):
        t0, t1 = edges[i], edges[i + 1]
        v0 = t0 - pd.Timedelta(days=VALID_DAYS)
        ef, dt = df.ev_first, df.date
        tr = ((ef < v0) & (dt < v0)).values
        va = ((ef >= v0) & (ef < t0) & (dt >= v0) & (dt < t0)).values
        te = ((ef >= t0) & (ef < t1) & (dt >= t0) & (dt < t1)).values
        if te.sum() < 500 or tr.sum() < 5000:
            continue
        yield (t0.strftime("%Y-%m"), tr, va, te)


def apply_null(df, mode, seed):
    """Destroy the language, hold category and price level fixed.

    The question is permuted within (month x category x price decile), so a
    shuffled row still carries a plausible category and a plausible price for its
    text -- only the sentence is wrong. Anything the text block can still explain
    after this is not language.

    Done as one vectorised gather rather than a per-group assignment: the key has
    thousands of cells on this panel and a groupby-and-assign loop over them costs
    more than the model fit it exists to falsify.
    """
    df = df.copy()
    df["market_id_txt"] = df.market_id
    if mode != "shuffle":
        return df
    rng = np.random.default_rng(seed)
    pdec = pd.qcut(df.p, 10, labels=False, duplicates="drop")
    mo = df.date.dt.to_period("M").astype(str)
    g = pd.DataFrame({"mo": mo, "cat": df.cat, "pdec": pdec}) \
          .groupby(["mo", "cat", "pdec"], observed=True).ngroup().values
    pos = np.arange(len(df))
    base = np.lexsort((pos, g))                    # positions grouped, stable
    order = np.lexsort((rng.random(len(df)), g))   # same groups, random within
    perm = np.empty(len(df), dtype=np.int64)
    perm[base] = order
    cols = ["text", "text_scrub", "market_id_txt"]
    df[cols] = df[cols].to_numpy()[perm]
    return df


# ------------------------------------------------------------------- features
def price_block(df, cats):
    x = pd.DataFrame(index=df.index)
    x["p"] = df.p
    x["logit_p"] = logit(df.p)
    x["p2"] = df.p ** 2
    x["logit2"] = logit(df.p) ** 2
    x["log_h2c"] = np.log1p(df.hours_to_close.clip(0, 1e5))
    x["log_age"] = np.log1p(df.age_days.clip(0))
    x["log_cum"] = np.log1p(df.gross_cum)
    x["log_day"] = np.log1p(df.gross_day)
    x["imb"] = (df.sgn_day / df.abs_day.replace(0, np.nan)).fillna(0)
    x["nbars"] = df.nbars
    x["neg"] = df.neg_risk.astype(float)
    for c in cats:
        x[f"cat_{c}"] = (df.cat == c).astype(float)
    return x.values.astype(np.float64)


def text_block(df, kind, field, tr):
    if kind == "none":
        return None, "none"
    if kind == "tfidf":
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True,
                            max_features=60000)
        v.fit(df.loc[tr, field])
        return v.transform(df[field]).astype(np.float64), f"tfidf[{len(v.vocabulary_)}]"
    z = np.load(os.path.join(NLP, f"emb_{kind}_{field}.npz"))
    lut = pd.Series(range(len(z["market_id"])), index=z["market_id"])
    E = z["emb"][lut.reindex(df.market_id_txt).values].astype(np.float64)
    return E, f"{kind}[{E.shape[1]}]"


def zscore(X, tr):
    if sparse.issparse(X):
        return X                      # tf-idf rows are already unit-norm
    mu, sd = X[tr].mean(0), X[tr].std(0)
    return (X - mu) / np.where(sd < 1e-9, 1.0, sd)


def rows(X, m):
    return X[np.where(m)[0]] if sparse.issparse(X) else X[m]


def stack(A, B, gamma):
    if B is None or gamma == 0.0:
        return A
    if sparse.issparse(B):
        return sparse.hstack([sparse.csr_matrix(A), B * gamma], format="csr")
    return np.hstack([A, B * gamma])


def best_ridge(make_X, y, w, tr, va, ev_va, grid):
    """Pick (alpha, gamma) on validation IC -- the statistic the signal is used
    through, and the one sec.4 warns is being selected on."""
    best = (-9e9, None, None, None)
    for g in grid:
        X = make_X(g)
        Xtr, Xva = rows(X, tr), rows(X, va)
        for a in ALPHAS:
            m = Ridge(alpha=a, fit_intercept=True,
                      solver="lsqr" if sparse.issparse(Xtr) else "auto")
            m.fit(Xtr, y[tr], sample_weight=w[tr])
            s = ic(m.predict(Xva), y[va], ev_va)["ic"]
            if s > best[0]:
                best = (s, a, g, m)
    return best


# ---------------------------------------------------------------------- book
def book_rows(pred, d, q=0.8, same_bar=False):
    """Long the top quintile of the signal, short the bottom, one share each.

    The signal is formed at the close of day t. Execution is therefore at day
    t+1's VWAPs, never at day t's own -- entering in the bar whose last print and
    volume produced the signal is a look-ahead, and on this panel it is worth
    about a third of the reported edge. `same_bar=True` reproduces that mistake
    on purpose, so the size of it can be quoted rather than asserted.

    Gross prices the same trip at the entry day's *mid* VWAP, net at the side a
    taker must actually hit, so the difference between them is exactly the half-
    spread and nothing else. Benchmarking gross against the day's last print
    instead -- the obvious choice -- silently mixes in the intraday drift between
    the VWAP and the close, which on this panel runs in the signal's favour and
    produces the nonsense of a net edge larger than its own gross. sec.3.5 makes
    the same choice for the same reason.

    The half-spread is floored at zero. On 7.7 % of market-days the day's
    buy-VWAP sits *below* its sell-VWAP -- not a crossed book but an artifact of
    volume-weighting a trending day, since the two sides are averaged over
    different hours. Taking those at face value would let the strategy be paid
    to cross, which is why net came out above gross before the floor. Flooring
    charges zero there and the true half-spread everywhere else, so the cost is
    never negative and the fill is never better than mid.

    Holding to resolution means only the entry crosses: the contract settles at
    par, so unlike sec.3.5's two-legged round trip there is no exit half-spread.

    Execution at a day's VWAP assumes a trader patient enough to work the order
    across the session. That is the assumption this venue's liquidity forces --
    sec.3.5 shows a single-bar taker crossing is 4.2x under water -- and it is
    stated rather than hidden.
    """
    hi, lo = np.quantile(pred, q), np.quantile(pred, 1 - q)
    side = np.where(pred >= hi, 1.0, np.where(pred <= lo, -1.0, 0.0))
    m = side != 0
    if m.sum() < 20:
        return None
    dd, s = d[m], side[m]
    if same_bar:
        buy, sell = dd.vwap_buy, dd.vwap_sell
    else:
        buy, sell = dd.vwap_buy_next, dd.vwap_sell_next
    ok = buy.notna() & sell.notna()
    if ok.sum() < 20:
        return None
    dd, s = dd[ok.values], s[ok.values]
    buy, sell = buy[ok.values].values, sell[ok.values].values
    ref = (buy + sell) / 2.0
    half = np.maximum((buy - sell) / 2.0, 0.0)
    ex = ref + np.where(s > 0, half, -half)
    return pd.DataFrame({"gross": s * (dd.win_event - ref),
                         "net": s * (dd.win_event - ex),
                         "event_id": dd.event_id.values, "w": dd.w.values})


def book_stats(parts):
    if not parts:
        return None
    b = pd.concat(parts, ignore_index=True)
    g = event_stats(b.gross, b.event_id, b.w)
    n = event_stats(b.net, b.event_id, b.w)
    return dict(trips=int(len(b)), n_events=g["n_events"],
                gross_pp=g["mean_pp"], t_gross=g["t"],
                net_pp=n["mean_pp"], t_net=n["t"],
                cost_pp=g["mean_pp"] - n["mean_pp"])


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="tfidf",
                    choices=["none", "tfidf", "finbert", "bert", "minilm"])
    ap.add_argument("--field", default="text", choices=["text", "text_scrub"])
    ap.add_argument("--kind", default="nl", choices=["nl", "date", "ts", "all"])
    ap.add_argument("--target", default="term", choices=["term", "fwd"])
    ap.add_argument("--split", default="walk",
                    choices=["walk", "random", "eventrandom"])
    ap.add_argument("--null", default="none", choices=["none", "shuffle"])
    ap.add_argument("--residual", action="store_true",
                    help="fit the text block on M1's residual, and book it alone")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    df = load(a.kind, a.target)
    df = apply_null(df, a.null, a.seed)
    cats = df.cat.value_counts().head(20).index.tolist()
    Xp_all = price_block(df, cats)
    y, w, ev = df.y.values, df.w.values, df.event_id.values

    fold_rows, books = [], {"M1": [], "M2": [], "TXT": []}
    tname = "none"
    for name, tr, va, te in make_folds(df, a.split, a.folds, a.seed):
        if va.sum() < 200:
            continue
        Xp = zscore(Xp_all, tr)
        Xt, tname = text_block(df, a.text, a.field, tr)
        if Xt is not None and not sparse.issparse(Xt):
            Xt = zscore(Xt, tr)

        v1, al1, _, m1 = best_ridge(lambda g: Xp, y, w, tr, va, ev[va], [0.0])
        p1 = m1.predict(Xp[te])
        r1 = ic(p1, y[te], ev[te])
        row = dict(fold=name, n_tr=int(tr.sum()), n_te=int(te.sum()),
                   ev_tr=int(pd.unique(ev[tr]).size), ev_te=int(pd.unique(ev[te]).size),
                   alpha1=al1, valid_ic1=v1, ic1=r1["ic"], t1=r1["ic_t"])
        books["M1"].append(book_rows(p1, df[te]))

        if Xt is not None:
            if a.residual:
                resid = y - m1.predict(Xp)
                v2, al2, g2, m2 = best_ridge(lambda g: Xt, resid, w, tr, va, ev[va], [1.0])
                ptxt = m2.predict(rows(Xt, te))
                p2 = p1 + ptxt
                books["TXT"].append(book_rows(ptxt, df[te]))
            else:
                v2, al2, g2, m2 = best_ridge(lambda g: stack(Xp, Xt, g), y, w,
                                             tr, va, ev[va], GAMMAS)
                p2 = m2.predict(rows(stack(Xp, Xt, g2), te))
            r2 = ic(p2, y[te], ev[te])
            row.update(alpha2=al2, gamma2=g2, valid_ic2=v2, ic2=r2["ic"], t2=r2["ic_t"],
                       dic=r2["ic"] - r1["ic"])
            books["M2"].append(book_rows(p2, df[te]))
        fold_rows.append(row)
        print(f"  fold {name}: tr {tr.sum():>7,} te {te.sum():>6,}  "
              f"IC1 {row['ic1']:+.4f}" +
              (f"  IC2 {row['ic2']:+.4f}  dIC {row['dic']:+.4f}"
               if 'ic2' in row else ""), flush=True)

    F = pd.DataFrame(fold_rows)
    if F.empty:
        print("no usable folds"); return
    pd.set_option("display.width", 220)
    print(f"panel {len(df):,} rows  {df.market_id.nunique():,} markets  "
          f"{df.event_id.nunique():,} events   text={tname}   split={a.split}"
          f"{'  NULL=shuffle' if a.null!='none' else ''}"
          f"{'  RESIDUAL' if a.residual else ''}")
    cols = [c for c in ["fold","n_tr","n_te","ev_te","alpha1","ic1","t1",
                        "alpha2","gamma2","ic2","t2","dic"] if c in F]
    print(F[cols].round(4).to_string(index=False))

    res = {"args": vars(a), "text_block": tname, "folds": F.to_dict("records")}
    print()
    print(f"pooled over {len(F)} folds:  IC(M1) {F.ic1.mean():+.4f}")
    res["ic1_mean"] = float(F.ic1.mean())
    if "dic" in F:
        n = len(F)
        t = F.dic.mean() / (F.dic.std(ddof=1) / np.sqrt(n)) if n > 2 else float("nan")
        res.update(ic2_mean=float(F.ic2.mean()), dic_mean=float(F.dic.mean()),
                   dic_t=float(t), dic_win=int((F.dic > 0).sum()), dic_n=n)
        print(f"                          IC(M2) {F.ic2.mean():+.4f}")
        print(f"                          dIC    {F.dic.mean():+.4f}  "
              f"(t over folds {t:+.2f}, positive in {int((F.dic>0).sum())}/{n})")
    for k in ("M1", "M2", "TXT"):
        b = book_stats([x for x in books[k] if x is not None])
        if b:
            res[f"book_{k}"] = b
            print(f"  {k:3s} book: {b['trips']:>7,} trips / {b['n_events']:>5,} events   "
                  f"gross {b['gross_pp']:+7.4f} pp (t={b['t_gross']:+6.2f})   "
                  f"net {b['net_pp']:+7.4f} pp (t={b['t_net']:+6.2f})   "
                  f"cost {b['cost_pp']:.4f}")

    tag = a.tag or "_".join([a.text, a.field, a.kind, a.target, a.split, a.null] +
                            (["resid"] if a.residual else []))
    out = os.path.join(NLP, f"signal_{tag}.json")
    json.dump(res, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
