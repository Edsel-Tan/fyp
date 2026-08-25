#!/usr/bin/env python3
"""Aggregate sweep results into the paper's Table 1 format and compare."""
import glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

# Zhao & Welsch (2024), arXiv:2410.14927v1, Table 1. S&P 500 universe, 10 seeds.
PAPER = {
    "test2021": {
        "HRT-FR-orig": dict(cum=0.3147, ann=0.3200, vol=0.1489, sharpe=2.1494, mdd=-0.0738),
        "PPO":         dict(cum=0.3428, ann=0.3486, vol=0.1498, sharpe=2.3274, mdd=-0.0808),
        "DDPG":        dict(cum=0.3813, ann=0.3879, vol=0.1458, sharpe=2.6601, mdd=-0.0651),
        "HRT-FR":      dict(cum=0.3983, ann=0.4051, vol=0.1665, sharpe=2.4336, mdd=-0.0845),
        "HRT":         dict(cum=0.4548, ann=0.4628, vol=0.1687, sharpe=2.7440, mdd=-0.0755),
        "Min-Var":     dict(cum=0.2368, ann=0.2406, vol=0.0980, sharpe=2.4549, mdd=-0.0516),
        "S&P500":      dict(cum=0.2913, ann=0.2961, vol=0.1303, sharpe=2.2736, mdd=-0.0521),
    },
    "test2022": {
        "HRT-FR-orig": dict(cum=-0.0154, ann=-0.0156, vol=0.0586, sharpe=-0.2664, mdd=-0.0448),
        "PPO":         dict(cum=-0.0045, ann=-0.0045, vol=0.0646, sharpe=-0.0701, mdd=-0.0431),
        "DDPG":        dict(cum=0.0174, ann=0.0176, vol=0.0790, sharpe=0.2224, mdd=-0.0484),
        "HRT-FR":      dict(cum=0.0229, ann=0.0232, vol=0.0893, sharpe=0.2594, mdd=-0.0554),
        "HRT":         dict(cum=0.0368, ann=0.0372, vol=0.0901, sharpe=0.4132, mdd=-0.0548),
        "Min-Var":     dict(cum=-0.0696, ann=-0.0704, vol=0.1513, sharpe=-0.4654, mdd=-0.1507),
        "S&P500":      dict(cum=-0.1995, ann=-0.2016, vol=0.2416, sharpe=-0.8344, mdd=-0.2543),
    },
}

LABEL = {
    "hrt_causal":    "HRT-FR  causal signal, \u03b1 per step",
    "hrt_causal_ep": "HRT-FR  causal signal, \u03b1 per episode",
    "hrt_paper":     "HRT-FR  paper timing, \u03b1 per step",
    "hrt_paper_ep":  "HRT-FR  paper timing, \u03b1 per episode",
    "ppo_causal":    "PPO (standalone)",
    "ddpg_causal":   "DDPG (standalone)",
}
PAPER_KEY = {"hrt_causal": "HRT-FR", "hrt_causal_ep": "HRT-FR",
             "hrt_paper": "HRT-FR", "hrt_paper_ep": "HRT-FR",
             "ppo_causal": "PPO", "ddpg_causal": "DDPG"}
KEYS = [("cum", "cum_return"), ("ann", "ann_return"), ("vol", "ann_vol"),
        ("sharpe", "sharpe"), ("mdd", "max_drawdown")]


def load_runs():
    """Key on the filename stem: agent_signal[_tag], with the seed suffix removed.
    Keying on agent+signal alone would merge the alpha-decay variants, which differ
    only by --tag."""
    runs = {}
    for f in sorted(glob.glob(os.path.join(ART, "runs", "*.json"))):
        stem = os.path.basename(f)[:-5]
        key = re.sub(r"_s\d+$", "", stem)
        if "_smoke" in key:
            continue
        runs.setdefault(key, []).append(json.load(open(f)))
    return runs


def agg(rs, period):
    out = {}
    for short, key in KEYS:
        v = np.array([r[period][key] for r in rs])
        out[short] = (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    # n_trades counts *requested* orders, so use executed notional instead:
    # turnover = executed notional / portfolio value, and cost_paid is exact.
    for extra in ("turnover", "cost_paid", "active_names_per_day",
                  "names_ever_traded", "trade_concentration_hhi"):
        v = np.array([r[period][extra] for r in rs if extra in r[period]], float)
        out[extra] = ((float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
                      if len(v) else (float("nan"), 0.0))
    out["n_seeds"] = len(rs)
    return out


def fmt(mean, sd, n):
    return f"{mean:+.4f}" + (f"±{sd:.3f}" if n > 1 else "")


def main():
    runs = load_runs()
    base = json.load(open(os.path.join(ART, "baselines.json")))
    if not runs:
        print("no runs yet"); return
    print(f"runs found: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(runs.items())))

    summary = {}
    for period, year in (("test2021", "2021 (bullish)"), ("test2022", "2022 (bearish)")):
        print(f"\n{'='*136}\n{year}\n{'='*136}")
        hdr = f"{'strategy':<44}{'cum ret':>16}{'ann ret':>16}{'ann vol':>16}{'sharpe':>16}{'max DD':>16}{'paper sharpe':>14}"
        print(hdr); print("-" * 136)
        rows = {}
        for arm in ("hrt_causal", "hrt_causal_ep", "hrt_paper", "hrt_paper_ep",
                    "ppo_causal", "ddpg_causal"):
            if arm not in runs:
                continue
            a = agg(runs[arm], period)
            rows[arm] = a
            n = a["n_seeds"]
            paper_key = PAPER_KEY[arm]
            ps = PAPER[period][paper_key]["sharpe"]
            print(f"{LABEL[arm]+f' [n={n}]':<44}"
                  f"{fmt(*a['cum'], n):>16}{fmt(*a['ann'], n):>16}{fmt(*a['vol'], n):>16}"
                  f"{fmt(*a['sharpe'], n):>16}{fmt(*a['mdd'], n):>16}{ps:>14.4f}")
        print("-" * 136)
        for name, key in (("Min-Variance (daily)", "min_variance"),
                          ("Min-Variance (net cost)", "min_variance_net"),
                          ("Equal-weight buy&hold", "equal_weight_bh"),
                          ("S&P 500 (^GSPC)", "sp500")):
            m = base[period][key]
            pk = {"min_variance": "Min-Var", "sp500": "S&P500"}.get(key)
            ps = f"{PAPER[period][pk]['sharpe']:.4f}" if pk else ""
            print(f"{name:<44}{m['cum_return']:>+16.4f}{m['ann_return']:>+16.4f}"
                  f"{m['ann_vol']:>16.4f}{m['sharpe']:>+16.4f}{m['max_drawdown']:>+16.4f}{ps:>14}")
            rows[key] = m
        summary[period] = {k: v for k, v in rows.items() if k != "curves"}

        # mean equity curve per arm, rebased to 1.0, for the figures
        curves = {}
        for arm in runs:
            V = np.array([r[period]["values"] for r in runs[arm]], float)
            curves[LABEL[arm]] = list((V / V[:, :1]).mean(0))
        for name, key in (("Min-Variance (daily)", "min_variance"),
                          ("Equal-weight buy&hold", "equal_weight_bh"),
                          ("S&P 500 (^GSPC)", "sp500")):
            v = np.array(base[period]["series"][key], float)
            curves[name] = list(v / v[0])
        summary[period]["_curves"] = curves

    print(f"\n{'='*136}\nturnover / activity\n{'='*136}")
    for period in ("test2021", "test2022"):
        for arm in sorted(runs):
            a = agg(runs[arm], period)
            act = a["active_names_per_day"][0]
            act_s = f"{act:6.1f}" if act == act else "   n/a"
            print(f"{period}  {LABEL[arm]:<44} daily turnover {a['turnover'][0]:.4f}"
                  f"  cost paid ${a['cost_paid'][0]:>10,.0f}"
                  f"  names traded/day {act_s}")
    json.dump(summary, open(os.path.join(ART, "summary.json"), "w"), indent=1)
    print("\nwrote", os.path.join(ART, "summary.json"))


if __name__ == "__main__":
    main()
