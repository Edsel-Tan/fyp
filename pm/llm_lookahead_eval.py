#!/usr/bin/env python3
"""Score pm/llm_lookahead.py's forecasts: is an LLM's edge over the market a memory?

Three measurements, each designed so that a forecasting model and a remembering
model predict different things:

  1. Edge over the market, by where a market resolved relative to the model's
     documented training cutoff. The edge is the LLM's *incremental* information:
     a logistic regression of the outcome on the market's own logit plus the LLM's,
     whose LLM coefficient is zero for a model that knows nothing the price does
     not. Standard errors are clustered on event. A forecaster's coefficient should
     not care about the cutoff; a memoriser's should be large before it and ~0 after.
  2. The cutoff, estimated blind. For every candidate month c, fit the same
     regression with separate LLM slopes before and after c and keep the c with the
     best likelihood. If the edge is memory, the estimated c lands on the documented
     cutoff without being told it -- and for the models whose cutoff is not
     documented, it is the only estimate there is.
  3. The same markets, two models. Markets that resolved between two models'
     cutoffs are *after* one and *before* the other, so any market-level confound
     (harder questions, a different era's categories) is held fixed. Memory predicts
     the later-cutoff model has the edge there and not on markets after both.

    python pm/llm_lookahead_eval.py
"""
import glob, json, os
import numpy as np, pandas as pd
import statsmodels.api as sm

LLM_EPS = 1e-12
HERE = os.path.dirname(os.path.abspath(__file__))
LLM = os.path.join(HERE, os.pardir, "data", "llm")

# Documented training-data cutoffs (model cards / technical reports), first of month.
# None = not documented; the blind estimate in (2) is then the only one there is.
CUTOFF = {
    "unsloth/Meta-Llama-3.1-8B-Instruct": "2023-12",     # Meta model card: December 2023
    "unsloth/gemma-3-12b-it": "2024-08",                 # Gemma 3 card: August 2024
    "allenai/Olmo-3-7B-Instruct": "2024-12",             # card: "Date cutoff: Dec. 2024"
    "google/gemma-4-12B-it": "2025-01",                  # card: "cutoff date of January 2025"
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": None,
    "Qwen/Qwen3.5-9B": None,
}


def logit(p, eps=1e-3):
    """eps = 1e-3 suits market prices (all in [0.03, 0.97] here). LLM answers are
    not: instruct models put P(Yes) below 1e-3 on ~71 % of these questions, so the
    LLM's log-odds are taken with LLM_EPS, or most of its ranking collapses onto the
    clip and the regression measures a constant."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def fit(d, cols):
    X = sm.add_constant(d[cols].to_numpy())
    try:
        r = sm.Logit(d.y.to_numpy(), X).fit(disp=0, cov_type="cluster",
                                             cov_kwds={"groups": d.event_id.to_numpy()})
    except Exception:
        return None
    return r


def edge(d):
    """LLM coefficient, its clustered t, and the log-loss gain over price alone."""
    if len(d) < 50 or d.event_id.nunique() < 30:
        return dict(n=len(d), events=int(d.event_id.nunique()))
    r0, r1 = fit(d, ["lp"]), fit(d, ["lp", "ll"])
    if r0 is None or r1 is None:
        return dict(n=len(d))
    bm = np.mean((d.p - d.y) ** 2); bl = np.mean((d.p_llm - d.y) ** 2)
    auc = pd.Series(d.p_llm.values).rank().pipe(
        lambda r: (r[d.y.values == 1].sum() - d.y.sum() * (d.y.sum() + 1) / 2)
        / max(d.y.sum() * (len(d) - d.y.sum()), 1))
    return dict(n=len(d), events=int(d.event_id.nunique()),
                beta_llm=float(r1.params[2]), t_llm=float(r1.tvalues[2]),
                dll_per_mkt=float((r1.llf - r0.llf) / len(d)),
                brier_mkt=float(bm), brier_llm=float(bl), auc_llm=float(auc))


def months_between(a, b):
    return (a.dt.year - b.year) * 12 + (a.dt.month - b.month)


def blind_cutoff(d, lo="2023-06", hi="2026-03"):
    """Month c maximising the likelihood of separate LLM slopes before and after c."""
    out = []
    for c in pd.period_range(lo, hi, freq="M"):
        pre = (d.res_month < c).astype(float)
        if pre.sum() < 100 or (1 - pre).sum() < 100:
            continue
        x = d.assign(ll_pre=d.ll * pre, ll_post=d.ll * (1 - pre), pre=pre)
        r = fit(x, ["lp", "pre", "ll_pre", "ll_post"])
        r0 = fit(x, ["lp", "pre", "ll"])              # same, one common LLM slope
        if r is not None and r0 is not None:
            out.append((str(c), float(r.llf), float(r.params[3]), float(r.params[4]),
                        float(2 * (r.llf - r0.llf))))
    if not out:
        return None
    best = max(out, key=lambda z: z[1])
    llf = np.array([z[1] for z in out])
    # a profile-likelihood 95% set: months within 1.92 log-lik units of the best
    ci = [z[0] for z in out if z[1] >= best[1] - 1.92]
    # Andrews (1993) sup-LR for a break at an unknown date: one restricted parameter,
    # ~15 % trimming -> 5 % critical value 8.85 (1 % 12.35). A best month without this
    # is just the argmax of noise, which is what the no-effect models return.
    sup = max(z[4] for z in out)
    return dict(month=best[0], beta_pre=best[2], beta_post=best[3],
                ci=[ci[0], ci[-1]], sup_lr=sup, sup_lr_crit5=8.85, break_significant=sup > 8.85,
                profile=[(z[0], round(z[1] - best[1], 2)) for z in out])


def main():
    m = pd.read_parquet(os.path.join(LLM, "markets.parquet"))
    m["res_date"] = pd.to_datetime(m.resolved_ts, unit="s")
    m["res_month"] = m.res_date.dt.to_period("M")
    files = sorted(glob.glob(os.path.join(LLM, "p_*.parquet")))
    P = {}
    for f in files:
        x = pd.read_parquet(f)
        prompt = x.prompt.iloc[0] if "prompt" in x else "forecast"
        P[(x.model.iloc[0], prompt)] = x.set_index("market_id").p_llm
    out = {"n_markets": len(m), "models": {}}
    bins = [-999, -12, -6, 0, 3, 6, 12, 999]
    labels = ["<=-12", "-12..-6", "-6..0", "0..3", "3..6", "6..12", ">12"]
    for (name, prompt), p_llm in sorted(P.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        d = m.join(p_llm, on="market_id").dropna(subset=["p_llm"]).copy()
        d["lp"], d["ll"] = logit(d.p), logit(d.p_llm, LLM_EPS)
        rec = {"model": name, "prompt": prompt, "cutoff": CUTOFF.get(name), "all": edge(d)}
        c = CUTOFF.get(name)
        if c:
            cp = pd.Period(c, "M")
            rel = (d.res_month - cp).apply(lambda z: z.n)
            d["rel"] = pd.cut(rel, bins, labels=labels, right=False)
            rec["pre"] = edge(d[rel < 0]); rec["post"] = edge(d[rel >= 0])
            rec["by_rel"] = {str(k): edge(g) for k, g in d.groupby("rel", observed=True)}
        rec["blind"] = blind_cutoff(d)
        out["models"][f"{name} [{prompt}]"] = rec
        a, b = rec.get("pre", {}), rec.get("post", {})
        print(f"\n{name} [{prompt}]  cutoff {c}")
        print(f"  all   {rec['all']}")
        if c:
            print(f"  pre   beta {a.get('beta_llm', np.nan):+.3f} t {a.get('t_llm', np.nan):+.2f} n {a.get('n')}")
            print(f"  post  beta {b.get('beta_llm', np.nan):+.3f} t {b.get('t_llm', np.nan):+.2f} n {b.get('n')}")
            for k, v in rec["by_rel"].items():
                print(f"    {k:>8}  beta {v.get('beta_llm', np.nan):+.3f}  t {v.get('t_llm', np.nan):+6.2f}"
                      f"  n {v.get('n'):>5}  brier llm/mkt {v.get('brier_llm', np.nan):.3f}/"
                      f"{v.get('brier_mkt', np.nan):.3f}")
        if rec["blind"]:
            bl = rec["blind"]
            print(f"  blind cutoff {bl['month']} (95% set {bl['ci'][0]}..{bl['ci'][1]})"
                  f"  beta pre {bl['beta_pre']:+.3f} post {bl['beta_post']:+.3f}"
                  f"  sup-LR {bl['sup_lr']:.2f} ({'break' if bl['break_significant'] else 'no break'} at 5%)")

    # (3) paired: same markets, earlier- vs later-cutoff model
    pairs = []
    for prompt in sorted({k[1] for k in P}):
        doc = sorted([(c, n) for n, c in CUTOFF.items() if c and (n, prompt) in P])
        for i in range(len(doc)):
            for j in range(i + 1, len(doc)):
                (ce, ne), (cl, nl) = doc[i], doc[j]
                if ce == cl:
                    continue
                d = m.join(P[(ne, prompt)].rename("pe"), on="market_id") \
                     .join(P[(nl, prompt)].rename("pl"), on="market_id").dropna()
                d["lp"] = logit(d.p)
                between = (d.res_month >= pd.Period(ce, "M")) & (d.res_month < pd.Period(cl, "M"))
                after = d.res_month >= pd.Period(cl, "M")
                row = dict(prompt=prompt, early=ne, late=nl, early_cutoff=ce, late_cutoff=cl)
                for lab, mask in (("between", between), ("after_both", after)):
                    g = d[mask]
                    if len(g) < 50:
                        row[lab] = dict(n=int(len(g))); continue
                    re_ = fit(g.assign(ll=logit(g.pe, LLM_EPS)), ["lp", "ll"])
                    rl = fit(g.assign(ll=logit(g.pl, LLM_EPS)), ["lp", "ll"])
                    row[lab] = dict(n=int(len(g)), beta_early=float(re_.params[2]), t_early=float(re_.tvalues[2]),
                                    beta_late=float(rl.params[2]), t_late=float(rl.tvalues[2]))
                pairs.append(row)
                print(f"\npair [{prompt}] {ne.split('/')[-1]} ({ce}) vs {nl.split('/')[-1]} ({cl})")
                for lab in ("between", "after_both"):
                    v = row[lab]
                    if "beta_early" in v:
                        print(f"  {lab:>10} n {v['n']:>5}  early beta {v['beta_early']:+.3f} (t {v['t_early']:+.2f})"
                              f"   late beta {v['beta_late']:+.3f} (t {v['t_late']:+.2f})")
    out["pairs"] = pairs
    json.dump(out, open(os.path.join(LLM, "lookahead_eval.json"), "w"), indent=1, default=str)
    print("\nwrote", os.path.join(LLM, "lookahead_eval.json"))


if __name__ == "__main__":
    main()
