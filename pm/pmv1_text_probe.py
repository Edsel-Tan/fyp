#!/usr/bin/env python3
"""What does a text model on this corpus actually read? ([16])

The nested regressions in `pmv1_text_signal.py` say the question adds nothing to
the price under an honest split, and a great deal under a careless one. This
script says *what* the careless version is reading, which is the part a reader
will not take on trust from a delta in the fourth decimal.

Two probes, both on the tf-idf block because it is the only one whose features
are legible:

  --probe terms   fit the text block on the terminal target under both splits and
                  print the n-grams carrying the largest weight. Under an i.i.d.
                  row split the heavy terms should be market identifiers -- proper
                  nouns, strike levels, the numeric disambiguators Polymarket
                  appends to a slug -- because the same market sits on both sides
                  and the cheapest way to predict its label is to name it. Under
                  the walk-forward split no such term can help.

  --probe clock   regress the observation's own calendar position on the text
                  alone. This quantifies the leak of RESULTS.md sec.3.6 directly:
                  if a slug string predicts *when* it traded, then any model given
                  that string in a temporally-interleaved split is being handed
                  the period, and its apparent skill is a date lookup.

    python pm/pmv1_text_probe.py --probe terms
    python pm/pmv1_text_probe.py --probe clock
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pmv1_text_signal as S


def top_terms(df, tr_mask, y, field, k=25):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
    Xtr = v.fit_transform(df.loc[tr_mask, field])
    m = Ridge(alpha=1.0, solver="lsqr").fit(Xtr, y[tr_mask.values],
                                            sample_weight=df.w.values[tr_mask.values])
    names = np.array(v.get_feature_names_out())
    c = m.coef_
    o = np.argsort(c)
    return pd.DataFrame({"term_pos": names[o[::-1][:k]], "w_pos": c[o[::-1][:k]],
                         "term_neg": names[o[:k]], "w_neg": c[o[:k]]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="terms", choices=["terms", "clock"])
    ap.add_argument("--kind", default="nl")
    ap.add_argument("--field", default="text")
    a = ap.parse_args()

    df = S.load(a.kind, "term")
    df = S.apply_null(df, "none", 0)
    pd.set_option("display.width", 200)

    if a.probe == "clock":
        # can the string alone place the row in time?
        t = (df.ts - df.ts.min()) / 86400.0
        rng = np.random.default_rng(0)
        u = rng.random(len(df))
        tr = pd.Series(u < 0.7, index=df.index)
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True)
        X = v.fit_transform(df.loc[tr, a.field])
        m = Ridge(alpha=1.0, solver="lsqr").fit(X, t[tr.values])
        pr = m.predict(v.transform(df.loc[~tr, a.field]))
        act = t[~tr.values].values
        r = np.corrcoef(pr, act)[0, 1]
        mae = np.abs(pr - act).mean()
        print(f"predicting the observation date from the slug alone ({a.field}):")
        print(f"  corr(pred, actual) = {r:+.4f}   R2 = {r**2:.4f}   "
              f"MAE = {mae:.1f} days   span = {act.max()-act.min():.0f} days")
        print(f"  naive (predict mean) MAE = {np.abs(act-act.mean()).mean():.1f} days")
        json.dump(dict(field=a.field, corr=float(r), r2=float(r ** 2), mae=float(mae)),
                  open(os.path.join(S.NLP, f"probe_clock_{a.field}.json"), "w"), indent=2)
        return

    y = df.y.values
    rng = np.random.default_rng(0)
    iid = pd.Series(rng.random(len(df)) < 0.7, index=df.index)
    cut = df.ev_first.quantile(0.7)
    wf = pd.Series((df.ev_first < cut) & (df.date < cut), index=df.index)

    print("=== heaviest tf-idf terms, i.i.d. row split (train rows include the "
          "same markets that appear in test) ===")
    print(top_terms(df, iid, y, a.field).to_string(index=False))
    print()
    print("=== heaviest tf-idf terms, walk-forward split (no market on both sides) ===")
    print(top_terms(df, wf, y, a.field).to_string(index=False))


if __name__ == "__main__":
    main()
