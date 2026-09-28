#!/usr/bin/env python3
"""Permutation null for pm/llm_lookahead_eval.py's blind-cutoff sup-LR.

Andrews' asymptotic 5 % value (8.85) assumes independent rows; markets here cluster
in events, and a 100-draw check put the empirical 95th percentile at ~10.5, with
12 % of null draws above 8.85. So each model's sup-LR is instead judged against its
own null: its log-odds permuted across markets *within the same resolution month*,
which keeps the model's output distribution and its month-to-month level but
destroys any market-level information, then the full blind search re-run.

    python pm/llm_lookahead_perm.py p_Qwen__Qwen3.5-9B.parquet 400
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_lookahead_eval as E

f, n = sys.argv[1], int(sys.argv[2])
m = pd.read_parquet(os.path.join(E.LLM, "markets.parquet"))
m["res_month"] = pd.to_datetime(m.resolved_ts, unit="s").dt.to_period("M")
x = pd.read_parquet(os.path.join(E.LLM, f)).set_index("market_id").p_llm
d = m.join(x, on="market_id").dropna(subset=["p_llm"]).copy()
d["lp"] = E.logit(d.p)
ll = E.logit(d.p_llm.values, E.LLM_EPS)
obs = E.blind_cutoff(d.assign(ll=ll))["sup_lr"]
groups = [np.flatnonzero(d.res_month.values == g) for g in d.res_month.unique()]
rng = np.random.default_rng(0)
null = []
for _ in range(n):
    perm = ll.copy()
    for idx in groups:
        perm[idx] = rng.permutation(perm[idx])
    null.append(E.blind_cutoff(d.assign(ll=perm))["sup_lr"])
null = np.array(null)
res = dict(file=f, n_perm=n, sup_lr=obs, p=float((1 + (null >= obs).sum()) / (1 + n)),
           null_q=dict(zip(["50", "90", "95", "99"], np.quantile(null, [.5, .9, .95, .99]).tolist())))
print(json.dumps(res))
json.dump(res, open(os.path.join(E.LLM, "perm_" + f.replace(".parquet", ".json")), "w"), indent=1)
