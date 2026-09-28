#!/usr/bin/env python3
"""Generate every figure in figs/ from the artifacts the analyses wrote.

Nothing here recomputes a result: each figure reads the same JSON/NPZ that
RESULTS.md and RAW.md quote, so a figure can never disagree with the table it
sits next to. Run the analyses first (RESULTS.md sec.8), then:

    python analysis/figures.py            # -> figs/*.png

Palette and chrome follow one convention throughout: a fixed categorical order
(never cycled), a single-hue ramp for magnitude, blue/red only where the encoding
is genuinely polar, hairline solid grid and axes, and direct labels on the marks
that carry the argument rather than a number on every point.
"""
import glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
ART = os.path.join(ROOT, "hrt", "artifacts")
CRY = os.path.join(ROOT, "crypto", "artifacts")
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figs")

# --- palette -----------------------------------------------------------------
# categorical slots, assigned in fixed order and never cycled
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = C
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.2,
})


def style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, axis=grid, alpha=1.0)
        ax.grid(False, axis="x" if grid == "y" else "y")
    return ax


def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(p, ROOT)}")


def note(fig, text):
    """Source line, so every figure says where its numbers came from."""
    fig.text(0.005, -0.02, text, fontsize=6.5, color=MUTED, ha="left", va="top")


# =============================================================================
def fig_falsification():
    """The audit: test IC on real vs martingale-difference panels, both timings."""
    sys.path.insert(0, os.path.join(ROOT, "hrt"))
    from falsify import ic_by_split
    res = {}
    for mode in ("causal", "paper"):
        real = ic_by_split(os.path.join(ART, f"fr_{mode}.npz"))["test"]
        nulls = []
        for f in sorted(glob.glob(os.path.join(ART, f"fr_synth_{mode}*.npz"))):
            nulls.append(ic_by_split(f)["test"])
        res[mode] = (real, np.array(nulls))

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.1),
                             gridspec_kw={"wspace": 0.52})
    zs = {"causal": "+2.20", "paper": "+28.5"}
    titles = {"causal": "Causal timing", "paper": "Paper's stated timing"}
    for ax, mode in zip(axes, ("causal", "paper")):
        real, nulls = res[mode]
        style(ax, grid="x")
        vals = list(nulls) + [real]
        names = [f"null panel #{i}" for i in range(len(nulls))] + ["REAL S&P 500"]
        col = [BLUE] * len(nulls) + [ORANGE]
        y = np.arange(len(vals))[::-1]
        ax.hlines(y, 0, vals, color=col, lw=1.5, zorder=2)
        ax.scatter(vals, y, s=[46] * len(nulls) + [96], color=col, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        ax.axvline(nulls.mean(), color=RED, lw=1.2, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_ylim(-0.85, len(vals) - 0.35)
        ax.set_title(titles[mode], loc="left", fontsize=9.5)
        ax.set_xlabel("test IC" if mode == "causal"
                      else "test IC   (note the changed scale)")
        # label only the mark that carries the argument, plus the null mean
        ax.annotate(f"{real:+.4f}", (real, 0), textcoords="offset points",
                    xytext=(9, 0), va="center", fontsize=9, color=INK, fontweight="bold")
        lo = min(0, min(vals))
        span = max(vals) - lo
        ax.set_xlim(lo - span * 0.16, max(vals) + span * 0.30)
        ax.annotate(f"null mean {nulls.mean():+.4f}", (nulls.mean(), -0.78),
                    textcoords="offset points", xytext=(4, 0), fontsize=7.5,
                    color=RED, va="bottom")
        ax.annotate(f"real vs null:  z = {zs[mode]}", (0.99, 1.0),
                    xycoords="axes fraction", ha="right", va="top", fontsize=8,
                    color=ORANGE if mode == "paper" else INK2, fontweight="bold")
    fig.suptitle("Falsification audit: the identical workflow on data with no predictability",
                 x=0.005, ha="left", y=1.10, fontsize=11, fontweight="bold", color=INK)
    fig.text(0.005, 1.015, "Under the paper's timing the pipeline reports IC +0.54 where "
             "nothing whatever is predictable.", fontsize=8.5, color=INK2, ha="left", va="top")
    note(fig, "Source: hrt/artifacts/fr_{causal,paper}.npz, fr_synth_*.npz via analysis/falsify.py. "
              "Null panels are martingale-difference returns matched on shape, calendar, volatility and factor loading.")
    save(fig, "fig1_falsification.png")


# =============================================================================
def fig_dsr():
    """Every leak-free arm sits below the Sharpe the search alone would produce."""
    J = {yr: json.load(open(os.path.join(ART, f"dsr_test{yr}.json")))
         for yr in ("2021", "2022")}
    R = {yr: {r["arm"]: r for r in J[yr]["arms"] if "[invalid]" not in r["arm"]}
         for yr in J}
    # one fixed arm order for BOTH panels -- a shared y-axis with per-panel sorting
    # would put each panel's rows against the other's labels
    order = sorted(R["2021"], key=lambda a: R["2021"][a]["sharpe_ann_repo"])
    thresh = {yr: J[yr]["sr_star_ann"] for yr in J}      # never hard-code: K grows
    K = J["2021"]["K"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    y = np.arange(len(order))
    for ax, yr in zip(axes, ("2021", "2022")):
        sh = [R[yr][a]["sharpe_ann_repo"] for a in order]
        col = [ORANGE if "paper" in a else BLUE for a in order]
        style(ax, grid="x")
        ax.hlines(y, 0, sh, color=col, lw=1.6, zorder=2)
        ax.scatter(sh, y, s=54, color=col, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.axvline(thresh[yr], color=RED, lw=1.4, zorder=4)
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel("annualised Sharpe (best seed)")
        ax.set_title(f"{yr} test year", loc="left")
        lo = min(0, min(sh))
        hi = max(thresh[yr], max(sh))
        ax.set_xlim(lo - (hi - lo) * 0.10, hi + (hi - lo) * 0.42)
        for i, a in enumerate(order):
            v = R[yr][a]["sharpe_ann_repo"]
            ax.annotate(f"DSR {R[yr][a]['dsr']:.3f}", (max(v, 0), i),
                        textcoords="offset points", xytext=(9, 0), va="center",
                        fontsize=7.5, color=INK2)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.annotate(f"SR*₀ = {thresh[yr]:.2f}", (thresh[yr], -0.62),
                    textcoords="offset points", xytext=(4, 0), fontsize=7.5,
                    color=RED, va="bottom")
    h = [plt.Line2D([], [], color=BLUE, marker="o", lw=1.6, ms=6, label="leak-free arms"),
         plt.Line2D([], [], color=ORANGE, marker="o", lw=1.6, ms=6,
                    label="leaky arms (paper timing)"),
         plt.Line2D([], [], color=RED, lw=1.4,
                    label=f"SR*₀ — expected max Sharpe from K = {K} trials")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.004, 1.045),
               ncol=3, fontsize=8)
    fig.suptitle("Deflated Sharpe: no leak-free arm beats what the search produces by luck",
                 x=0.004, ha="left", y=1.17, fontsize=11, fontweight="bold", color=INK)
    note(fig, "Source: hrt/artifacts/dsr_test{2021,2022}.json via analysis/dsr.py. "
              f"K = {K} counts every run ever aggregated, including runs_invalid/. Arms ordered by 2021 Sharpe in both panels.")
    save(fig, "fig2_deflated_sharpe.png")


# =============================================================================
def fig_cost_rerank():
    """MACE's claim: the cost model reorders the algorithms."""
    J = json.load(open(os.path.join(ART, "cost_rerank.json")))["test2021"]
    colour = lambda n: (ORANGE if "paper" in n else
                        VIOLET if J["turnover"][n] < 0.05 else BLUE)
    arms = {n: (J["turnover"][n], [J["sharpe"][n][k] for k in J["rates"]], colour(n))
            for n in sorted(J["sharpe"], key=lambda a: -J["sharpe"][a]["gross (0bp)"])}
    x = [int(round(v * 1e4)) for v in J["rates"].values()]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    style(ax)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for name, (turn, sh, col) in arms.items():
        dash = (0, (4, 2)) if "_ep" in name else "-"
        ax.plot(x, sh, color=col, lw=1.8, marker="o", ms=5, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.0, linestyle=dash)
    # end labels, pushed apart so six lines converging at 60 bp stay readable
    ends = sorted(((sh[-1], name, col) for name, (_, sh, col) in arms.items()),
                  reverse=True)
    gap, placed = 0.30, []
    for v, name, col in ends:
        t = v if not placed else min(v, placed[-1] - gap)
        placed.append(t)
        ax.annotate(f" {name}", (x[-1] + 1.5, t), fontsize=8, color=col,
                    va="center", fontweight="bold", annotation_clip=False)
        if abs(t - v) > 0.02:
            ax.plot([x[-1], x[-1] + 1.5], [v, t], color=col, lw=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xlabel("flat transaction cost charged (bp per unit turnover)")
    ax.set_ylabel("2021 annualised Sharpe")
    ax.set_xlim(-3, 84)
    ax.set_title("Cost re-ranking: the near-zero-turnover baselines climb as the rate rises",
                 loc="left")
    ax.annotate("rank changes vs the 10 bp default:\n"
                + "   ·   ".join(f"{m}/{J['n_arms']} at {k}" for k, m in J["moved"].items()),
                (0.015, 0.08), xycoords="axes fraction", fontsize=8, color=INK2)
    h = [plt.Line2D([], [], color=ORANGE, lw=1.8, label="hierarchy, paper timing (leaky)"),
         plt.Line2D([], [], color=BLUE, lw=1.8, label="hierarchy, causal timing"),
         plt.Line2D([], [], color=VIOLET, lw=1.8, label="flat baselines (turnover ≈ 0.004)")]
    ax.legend(handles=h, loc="upper right", bbox_to_anchor=(0.84, 1.0), fontsize=8)
    note(fig, "Source: analysis/cost_rerank.py, first-order recharge r_net(ρ) = r_net(10bp) + (0.001 − ρ)·turnover "
              "on hrt/artifacts/runs/. Used for ordering only; policies were not checkpointed. Dashed = episode-unit alpha.")
    save(fig, "fig3_cost_reranking.png")


# =============================================================================
def fig_censoring():
    """What a censored vendor calendar costs, measured against ground truth."""
    d = json.load(open(os.path.join(ROOT, "analysis", "censoring.json")))
    masks = ["gap geometry (9 on / 21 off)", "seasonal (hourly month retention)"]
    short = {masks[0]: "gap mask\n(36.8% retained)", masks[1]: "seasonal mask\n(65.7% retained)"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, run in zip(axes, ("mom20", "mom5")):
        R = d["runs"][run]
        style(ax)
        groups, w = np.arange(len(masks)), 0.22
        truth = R["truth"]["sharpe"]
        for j, (view, col) in enumerate((("naive", ORANGE), ("block", BLUE))):
            vals = [R[m][view]["sharpe"][0] for m in masks]
            errs = [R[m][view]["sharpe"][1] for m in masks]
            pos = groups + (j - 0.5) * (w + 0.03)
            ax.bar(pos, vals, w, color=col, zorder=3, linewidth=0,
                   label={"naive": "naive — gaps ignored",
                          "block": "block-aware"}[view])
            ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=INK2,
                        elinewidth=0.9, capsize=2.5, zorder=4)
            for x0, v, e in zip(pos, vals, errs):
                ax.annotate(f"{v:+.2f}", (x0, v + e), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=7.5, color=INK2)
        ax.axhline(truth, color=RED, lw=1.4, zorder=5)
        ax.annotate(f"truth  {truth:+.3f}", (len(masks) - 0.48, truth),
                    textcoords="offset points", xytext=(0, 5), fontsize=8,
                    color=RED, fontweight="bold", ha="right")
        ax.set_xticks(groups)
        ax.set_xticklabels([short[m] for m in masks], fontsize=8)
        ax.set_title({"mom20": "Momentum-20", "mom5": "Momentum-5"}[run], loc="left")
        ax.set_ylim(0, 1.62)
        ax.set_xlim(-0.55, len(masks) - 0.45)
    axes[0].set_ylabel("annualised Sharpe")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("A censored calendar doubles the reported Sharpe; blocking fixes the "
                 "mechanism, not the sample",
                 x=0.004, ha="left", y=1.04, fontsize=11, fontweight="bold", color=INK)
    note(fig, "Source: analysis/censoring.json (20 phase replicates, error bars = replicate sd). "
              "The crypto tape's own retention pattern applied in reverse to the complete 2,414-day × 370-name equity panel.")
    save(fig, "fig4_censoring_cost.png")


# =============================================================================
def fig_pooling():
    """Where the residual over-dispersion in the skill null comes from."""
    d = json.load(open(os.path.join(DATA, "pmv1_pooling.json")))
    lad = d["ladder"]
    lbl = {"trade = one fill (as published)": "trade  (as published)",
           "bet   = trader x market": "bet  = trader × market",
           "event = trader x neg-risk event": "event  = trader × neg-risk contract",
           "trader x close-day x category": "trader × close-day × category",
           "trader x close-day": "trader × close-day",
           "trader (degenerate: one flip)": "trader  (degenerate: one flip)"}
    y = np.arange(len(lad))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    style(ax, grid="x")
    skill = [p["sd_skill"] for p in d["persistence"]]
    ax.axvspan(min(skill), max(skill), color=SEQ[0], zorder=0)
    ax.axvline(1.0, color=RED, lw=1.4, zorder=4)
    for ser, key, col, m in (("rms(z)  — the null fixes this at 1.000", "rms_z", BLUE, "o"),
                             ("sd(z)  — as quoted in §3.1", "sd_z", ORANGE, "s")):
        v = [r[key] for r in lad]
        ax.plot(v, y, color=col, lw=1.4, marker=m, ms=6, zorder=3, label=ser,
                markeredgecolor=SURFACE, markeredgewidth=1.1)
    # label only the rungs the argument turns on
    for unit, dy in (("trade = one fill (as published)", 10),
                     ("event = trader x neg-risk event", 10),
                     ("trader x close-day", 10),
                     ("trader (degenerate: one flip)", 10)):
        r = next(x for x in lad if x["unit"] == unit)
        i = [x["unit"] for x in lad].index(unit)
        ax.annotate(f"{r['rms_z']:.3f}", (r["rms_z"], y[i]), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([lbl[r["unit"]] for r in lad], fontsize=8)
    ax.set_xlabel("calibration of the sign-randomisation null   (1.000 = correctly specified)")
    ax.set_xlim(0.84, 1.80)
    ax.set_ylim(-1.05, len(lad) - 0.25)
    ax.legend(loc="lower right", fontsize=7.5, bbox_to_anchor=(1.0, 0.02))
    ax.set_title("Coarsening the randomisation unit does not calibrate the null", loc="left")
    ax.annotate("what out-of-sample\npersistence alone implies\n(3 cut dates)",
                ((min(skill) + max(skill)) / 2, 4.55), ha="center", va="center",
                fontsize=7.5, color="#184f95")
    ax.annotate("the null requires\nrms(z) = 1.000", (1.0, -0.98),
                textcoords="offset points", xytext=(5, 0), fontsize=7.5,
                color=RED, va="bottom")
    ev = next(x for x in lad if x["unit"].startswith("event"))["rms_z"]
    po = next(x for x in lad if x["unit"] == "trader x close-day")["rms_z"]
    ax.annotate(f"maximal same-day pooling buys\nonly {ev - po:.3f} of the {ev - 1:.3f} to explain",
                (po, 1.0), xytext=(1.47, 1.75), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8), ha="left", va="center")
    note(fig, f"Source: data/pmv1_pooling.json via pm/pmv1_pooling.py — exact over all {d['n_accounts']:,} accounts with ≥10 resolved fills. "
              "Skill band from V = 2ρ/(1−ρ) on the three split-half correlations of pm/pmv1_persistence.py.")
    save(fig, "fig5_null_calibration.png")


# =============================================================================
def fig_persistence():
    """Skill persists: out-of-sample return by in-sample rank."""
    cuts = ["2024-07-01", "2025-01-01", "2025-07-01"]
    cols = [BLUE, ORANGE, VIOLET]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3),
                             gridspec_kw={"wspace": 0.38})
    ax = style(axes[0])
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    for cut, col in zip(cuts, cols):
        d = json.load(open(os.path.join(DATA, f"pmv1_persistence_{cut}.json")))
        r = [x["roi_out"] * 100 for x in d["deciles"]]
        ax.plot(range(1, 11), r, color=col, lw=1.6, marker="o", ms=4, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=f"cut {cut}   (n = {d['n_accounts']:,})")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("in-sample z decile")
    ax.set_ylabel("out-of-sample PnL / notional  (%)")
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(-2.9, 3.9)
    ax.set_title("Out-of-sample return by in-sample rank", loc="left", fontsize=9.5)
    ax.legend(loc="lower left", fontsize=7.5)
    ax.annotate("only the extreme deciles\norder reliably", (4.6, 2.7), fontsize=7.5,
                color=INK2, ha="center")

    ax = style(axes[1])
    lift = []
    for cut in cuts:
        d = json.load(open(os.path.join(DATA, f"pmv1_persistence_{cut}.json")))
        lift.append((cut, d["hit"] * 100, d["base"] * 100))
    yy = np.arange(len(lift))[::-1]
    for i, (cut, hit, base) in enumerate(lift):
        ax.plot([base, hit], [yy[i], yy[i]], color=AXIS, lw=1.4, zorder=2)
        ax.scatter([base], [yy[i]], s=54, color=MUTED, zorder=3,
                   edgecolor=SURFACE, linewidth=1.1)
        ax.scatter([hit], [yy[i]], s=54, color=BLUE, zorder=3,
                   edgecolor=SURFACE, linewidth=1.1)
        ax.annotate(f"{hit/base:.1f}×", ((base + hit) / 2, yy[i]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8.5, color=INK, fontweight="bold")
    # direct labels on the top row instead of a legend box
    _, hit0, base0 = lift[0]
    ax.annotate("base rate\n(all accounts)", (base0, yy[0]), textcoords="offset points",
                xytext=(0, -13), ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.annotate("flagged in\nsample", (hit0, yy[0]), textcoords="offset points",
                xytext=(0, -13), ha="center", va="top", fontsize=7.5, color=BLUE)
    ax.set_yticks(yy)
    ax.set_yticklabels([c for c, _, _ in lift], fontsize=8)
    ax.set_xlabel("flagged skilled out of sample  (%)")
    ax.set_ylim(-0.6, len(lift) - 0.15)
    ax.set_xlim(0, 50)
    ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.set_title("Flagged before the cut → flagged after", loc="left", fontsize=9.5)
    fig.suptitle("Prediction-market skill persists out of sample — a test that needs no null",
                 x=0.004, ha="left", y=1.06, fontsize=11, fontweight="bold", color=INK)
    note(fig, "Source: data/pmv1_persistence_*.json via pm/pmv1_persistence.py. Events are assigned whole, "
              "so the two halves share no resolution outcome; accounts need ≥10 fills in each half.")
    save(fig, "fig6_persistence.png")


# =============================================================================
def fig_imbalance():
    """[14]'s predictive coefficient, by how much notional the bar carries."""
    d = json.load(open(os.path.join(DATA, "pmv1_imbalance_1h.json")))
    qs = [f"liq_Q{i}" for i in range(1, 6)]
    med = [1, 13, 81, 394, 3569]
    beta = [d[q]["oi_large"][0] for q in qs]
    tval = [d[q]["oi_large"][2] for q in qs]
    se = [abs(b / t) if t else 0 for b, t in zip(beta, tval)]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    style(ax)
    col = [BLUE if b > 0 else RED for b in beta]
    x = np.arange(5)
    ax.bar(x, beta, 0.42, color=col, zorder=3, linewidth=0)
    ax.errorbar(x, beta, yerr=[2 * s for s in se], fmt="none", ecolor=INK2,
                elinewidth=0.9, capsize=3, zorder=4)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=2)
    for i, (b, t) in enumerate(zip(beta, tval)):
        se_i = 2 * se[i]
        ax.annotate(f"{b:+.4f}\nt {t:+.2f}", (i, b + (se_i if b > 0 else -se_i)),
                    textcoords="offset points", xytext=(0, 6 if b > 0 else -22),
                    ha="center", fontsize=7.5,
                    color=INK if abs(t) > 2 else MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}\nmedian ${m:,}" for i, m in enumerate(med)], fontsize=8)
    ax.set_ylabel("β, large-trade imbalance → next-bar return")
    ax.set_ylim(-0.0028, 0.0080)
    ax.axvspan(1.55, 2.45, color="#f4f3ee", zorder=0)
    ax.annotate("the fifteen-week vendor tape's\nmedian bar carried $69 — here",
                (2, 0.0031), ha="center", fontsize=7.5, color=ORANGE, fontweight="bold")
    ax.set_title("The imbalance effect lives entirely in the thick bars\n"
                 "5,244,127 hourly bars over 25,317 markets, by gross notional traded",
                 loc="left", fontsize=10)
    note(fig, "Source: data/pmv1_imbalance_1h.json via pm/pmv1_imbalance.py. Market fixed effects by within-transformation; "
              "standard errors clustered two-way on market and hour; error bars ±2 s.e.")
    save(fig, "fig7_imbalance_liquidity.png")


# =============================================================================
def fig_spread():
    """The same signal, priced at the prices a taker actually gets."""
    d = json.load(open(os.path.join(DATA, "pmv1_spread_1h.json")))
    rt, holds = d["roundtrip"], d["holds"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3),
                             gridspec_kw={"wspace": 0.30})

    ax = style(axes[0])
    keys = [f"liquidity Q{i}" for i in range(1, 6)]
    x, w = np.arange(5), 0.34
    g = [rt[k]["gross_pp"] for k in keys]
    n = [rt[k]["net_pp"] for k in keys]
    ax.bar(x - w / 2 - 0.01, g, w, color=BLUE, label="gross (mid-to-mid)", zorder=3, linewidth=0)
    ax.bar(x + w / 2 + 0.01, n, w, color=RED, label="net (at taker VWAPs)", zorder=3, linewidth=0)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([f"Q{i}" for i in range(1, 6)], fontsize=8)
    ax.set_xlabel("bar liquidity quintile")
    ax.set_ylabel("probability points per share")
    ax.set_title("One-bar round trip", loc="left", fontsize=9.5)
    ax.legend(loc="lower left", fontsize=7.5, bbox_to_anchor=(-0.02, -0.02))
    ax.set_ylim(-1.12, 0.22)
    ax.annotate(f"+{rt['liquidity Q5']['gross_pp']:.3f}", (4 - w / 2, g[4]),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5, color=BLUE)
    ax.annotate(f"{rt['liquidity Q5']['net_pp']:.3f}", (4 + w / 2, n[4]),
                textcoords="offset points", xytext=(0, -11), ha="center", fontsize=7.5, color=RED)

    ax = style(axes[1])
    hk = ["hold 1h", "hold 2h", "hold 4h", "hold 8h", "hold 24h"]
    hx = np.arange(len(hk))
    for key, col, m, lab in (("gross_pp", BLUE, "o", "gross edge"),
                             ("cost_pp", ORANGE, "s", "round-trip cost"),
                             ("net_pp", RED, "D", "net")):
        v = [holds[k][key] for k in hk]
        ax.plot(hx, v, color=col, lw=1.8, marker=m, ms=5, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.0)
        ax.annotate(f" {lab}", (hx[-1], v[-1]), fontsize=8, color=col,
                    va="center", fontweight="bold", annotation_clip=False)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=2)
    ax.set_xticks(hx); ax.set_xticklabels(["1 h", "2 h", "4 h", "8 h", "24 h"], fontsize=8)
    ax.set_xlim(-0.25, 5.9)
    ax.set_xlabel("holding period (top liquidity quintile)")
    ax.set_title("A longer hold does not amortise it", loc="left", fontsize=9.5)
    fig.suptitle("[14]'s effect is real and is worth nothing to the side that reads it",
                 x=0.005, ha="left", y=1.04, fontsize=11, fontweight="bold", color=INK)
    note(fig, "Source: data/pmv1_spread_1h.json via pm/pmv1_spread.py. Entry and exit priced at the VWAP a taker "
              "must hit; gross is the same trip mid-to-mid. fee_usdc is identically zero across the archive, so the cost is the spread.")
    save(fig, "fig8_spread_roundtrip.png")


# =============================================================================
def fig_crypto():
    """Crypto: a real gross edge, entirely consumed by turnover."""
    R = json.load(open(os.path.join(CRY, "strategy_costs.json")))
    strat = sorted({k.split("|", 1)[1] for k in R},
                   key=lambda s: -R[f"no cost|{s}"]["sharpe"])
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    style(ax, grid="x")
    y = np.arange(len(strat))[::-1]
    for i, s in enumerate(strat):
        a, b = R[f"no cost|{s}"]["sharpe"], R[f"flat 10bp|{s}"]["sharpe"]
        ax.plot([b, a], [y[i], y[i]], color=AXIS, lw=1.3, zorder=2)
        ax.scatter([a], [y[i]], s=50, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.1)
        ax.scatter([b], [y[i]], s=50, color=RED, zorder=3, edgecolor=SURFACE, linewidth=1.1)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.axvline(2.782, color=ORANGE, lw=1.4, zorder=4)
    ax.annotate("SR*₀ = 2.78\n(K = 11 trials)", (2.782, y[-1] - 0.2),
                textcoords="offset points", xytext=(5, 0), fontsize=7.5, color=ORANGE)
    ax.set_yticks(y)
    ax.set_yticklabels([s + f"   ({R['no cost|' + s]['turnover']:.3f})" for s in strat],
                       fontsize=7.5)
    ax.set_xlabel("annualised Sharpe over the 2026 out-of-sample window")
    ax.set_xlim(-23.5, 6.2)
    ax.set_ylim(-0.7, len(strat) - 0.3)
    h = [plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=6, label="gross of cost"),
         plt.Line2D([], [], color=RED, marker="o", ls="", ms=6, label="at a flat 10 bp")]
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(0.01, 0.60), fontsize=8)
    ax.set_title("Crypto: turnover eats the edge (strategy label shows turnover per bar)",
                 loc="left")
    note(fig, "Source: crypto/artifacts/strategy_costs.json via crypto/strategy.py. "
              "2,665 hourly bars × 17 coins over 4 contiguous blocks, 2026-03-08 → 2026-06-30. SR*₀ from crypto/dsr.py.")
    save(fig, "fig9_crypto_turnover.png")


# =============================================================================
def fig_text():
    """What a language model reads off a prediction-market question."""
    NLP = os.path.join(DATA, "nlp")

    def sig(tag):
        f = os.path.join(NLP, f"signal_{tag}.json")
        return json.load(open(f)) if os.path.exists(f) else None

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6),
                             gridspec_kw={"wspace": 0.42, "width_ratios": [1.05, 1.2, 0.85]})

    # --- left: the split convention is worth more than the model ------------
    ax = style(axes[0])
    splits = [("random\nrows", "all_term_random_none"),
              ("random\nevents", "all_term_eventrandom_none"),
              ("walk-\nforward", "all_term_walk_none")]
    x, w = np.arange(len(splits)), 0.36
    for i, (enc, col, lab) in enumerate((("tfidf", BLUE, "tf-idf"),
                                         ("finbert", ORANGE, "FinBERT"))):
        v = [(sig(f"{enc}_text_{t}") or {}).get("dic_mean", np.nan) for _, t in splits]
        xs = x + (i - 0.5) * (w + 0.02)
        ax.bar(xs, v, w, color=col, zorder=3, linewidth=0, label=lab)
        for k, (xi, vi) in enumerate(zip(xs, v)):
            if not np.isfinite(vi) or abs(vi) < 0.02:
                continue
            ax.annotate(f"{vi:+.3f}", (xi, vi), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=7.5, color=col)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in splits], fontsize=8)
    ax.set_ylabel("dIC from adding the question")
    ax.set_ylim(-0.055, 0.575)
    ax.set_title("The split decides the answer", loc="left", fontsize=9.5)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)
    ax.annotate("both ≈ 0", (2, 0.012), fontsize=7.5, color=MUTED, ha="center")

    # --- middle: no encoder clears the price baseline -----------------------
    ax = style(axes[1])
    encs = [("tf-idf", "tfidf"), ("FinBERT", "finbert"),
            ("BERT-base", "bert"), ("MiniLM", "minilm")]
    vals, errs, names = [], [], []
    for lab, e in encs:
        d = sig(f"{e}_text_nl_term_walk_none")
        if d is None:
            continue
        names.append(lab); vals.append(d.get("dic_mean", np.nan))
        dd = [r.get("dic", np.nan) for r in d.get("folds", [])]
        errs.append(np.nanstd(dd, ddof=1) / max(np.sqrt(len(dd)), 1) if dd else np.nan)
    y = np.arange(len(names))
    # the null arm: language destroyed, category and price level held
    nv = [d["dic_mean"] for d in
          (sig(f"{e}_text_nl_term_walk_shuffle") for _, e in encs)
          if d and "dic_mean" in d]
    if nv:
        lo, hi = min(nv), max(nv)
        pad = max((hi - lo) / 2, 0.0012)
        ax.axvspan(lo - pad, hi + pad, color=ORANGE, alpha=0.22, zorder=1, linewidth=0)
        ax.annotate("shuffled-question\nnull", (hi + pad, -0.72), fontsize=7.5,
                    color=ORANGE, ha="center", va="bottom", fontweight="bold",
                    annotation_clip=False)
    ax.barh(y, vals, 0.5, xerr=errs, color=BLUE, zorder=3, linewidth=0,
            error_kw=dict(ecolor=MUTED, elinewidth=0.9, capsize=2.5))
    ax.axvline(0, color=AXIS, lw=0.8, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_ylim(len(names) - 0.45, -1.35)
    ax.set_xlabel("dIC over price + category, walk-forward")
    ax.set_title("No encoder clears the baseline", loc="left", fontsize=9.5)

    # --- right: why -- the slug is a clock -----------------------------------
    ax = style(axes[2])
    ck = [("verbatim", "text"), ("dates, years,\nintegers masked", "text_scrub")]
    r2, labs = [], []
    for lab, f in ck:
        p_ = os.path.join(NLP, f"probe_clock_{f}.json")
        if os.path.exists(p_):
            r2.append(json.load(open(p_))["r2"]); labs.append(lab)
    xb = np.arange(len(labs))
    ax.bar(xb, r2, 0.46, color=SEQ[4], zorder=3, linewidth=0)
    for xi, vi in zip(xb, r2):
        ax.annotate(f"{vi:.3f}", (xi, vi), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8.5, color=SEQ[6], fontweight="bold")
    ax.set_xticks(xb); ax.set_xticklabels(labs, fontsize=8)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("R² predicting the trading date")
    ax.set_title("...the question is a timestamp", loc="left", fontsize=9.5)

    fig.suptitle("A language model on a prediction market reads the calendar, not the question",
                 x=0.005, ha="left", y=1.04, fontsize=11, fontweight="bold", color=INK)
    note(fig, "Source: data/nlp/signal_*.json and probe_clock_*.json via pm/pmv1_text_signal.py and "
              "pm/pmv1_text_probe.py. dIC is the walk-forward test IC of price+category+text minus that of "
              "price+category over 8 monthly folds; the text block carries a jointly-searched scale whose zero "
              "reproduces the baseline exactly. Error bars are the standard error over folds.")
    save(fig, "fig10_text_leak.png")


# =============================================================================
def fig_null_sweep():
    """The audit at the agent level: the RL sweep retrained where nothing is learnable."""
    J = json.load(open(os.path.join(ART, "null_sweep.json")))
    P = J["periods"]
    order = sorted(P["test2021"]["arms"],
                   key=lambda a: P["test2021"]["arms"][a]["real_exc"])

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3),
                             gridspec_kw={"wspace": 0.42, "width_ratios": [1, 1, 0.82]})
    y = np.arange(len(order))
    for ax, period, yr in zip(axes[:2], ("test2021", "test2022"), ("2021", "2022")):
        style(ax, grid="x")
        A = P[period]["arms"]
        nul = [A[a]["null_exc"] for a in order]
        rea = [A[a]["real_exc"] for a in order]
        col = [ORANGE if "paper" in a else BLUE for a in order]
        for i, (n, r, c) in enumerate(zip(nul, rea, col)):
            ax.plot([n, r], [y[i], y[i]], color=c, lw=1.2, alpha=0.45, zorder=2)
        ax.scatter(nul, y, s=52, facecolor=SURFACE, edgecolor=col, linewidth=1.6, zorder=3)
        ax.scatter(rea, y, s=52, color=col, zorder=4, edgecolor=SURFACE, linewidth=1.0)
        ax.axvline(0, color=RED, lw=1.2, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlabel("excess return over its own panel's floor")
        ax.set_title(f"{yr} test year", loc="left")
        lo, hi = min(nul + rea + [0]), max(nul + rea + [0])
        ax.set_xlim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.12)
    axes[0].annotate("floor", (0, -0.62), textcoords="offset points", xytext=(4, 0),
                     fontsize=7.5, color=RED, va="bottom")

    # --- right: the statistic the agent is checkpointed on -------------------
    ax = style(axes[2], grid="y")
    runs = {}
    for kind, d in (("null", "runs_null"), ("real", "runs")):
        for f in sorted(glob.glob(os.path.join(ART, d, "*.json"))):
            arm = os.path.basename(f)[:-5].rsplit("_s", 1)[0]
            if "_smoke" in f:
                continue
            k = ("leaky" if "paper" in arm else "leak-free", kind)
            runs.setdefault(k, []).append(json.load(open(f))["valid_sharpe"])
    cats = [("leak-free", "null"), ("leak-free", "real"), ("leaky", "null"), ("leaky", "real")]
    rng = np.random.default_rng(0)
    for i, c in enumerate(cats):
        v = np.array(runs[c])
        col = ORANGE if c[0] == "leaky" else BLUE
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=20, zorder=3,
                   facecolor=SURFACE if c[1] == "null" else col,
                   edgecolor=col, linewidth=1.1, alpha=0.9)
        ax.hlines(v.mean(), i - 0.30, i + 0.30, color=INK, lw=1.8, zorder=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["null", "real", "null", "real"], fontsize=8)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("validation Sharpe at checkpoint")
    ax.set_xlabel("the statistic train.py checkpoints on")
    for i, lab in ((0.5, "leak-free"), (2.5, "leaky")):
        ax.annotate(lab, (i, 1.012), xycoords=("data", "axes fraction"), ha="center",
                    va="bottom", fontsize=9, color=INK, fontweight="bold")
    lf = J["valid_sharpe"]["leak-free"]
    ax.axhline(lf["null_median"], xmin=0.03, xmax=0.47, color=RED, lw=1.0, ls=(0, (3, 2)), zorder=2)
    ax.annotate(f"{lf['real_below_null_median']} of {lf['real_n']} real runs\nsit below the null median",
                (1.34, lf["null_median"]), textcoords="offset points", xytext=(0, -9),
                fontsize=7.5, color=RED, ha="left", va="top")

    h = [plt.Line2D([], [], color=INK2, marker="o", ls="", ms=6, mfc=SURFACE, mew=1.4,
                    label="trained on a null panel"),
         plt.Line2D([], [], color=INK2, marker="o", ls="", ms=6, label="trained on the real panel"),
         plt.Line2D([], [], color=BLUE, lw=2.4, label="leak-free (causal timing)"),
         plt.Line2D([], [], color=ORANGE, lw=2.4, label="leaky (paper timing)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.004, 1.055), ncol=4, fontsize=8)
    fig.suptitle("The falsification audit at the agent level: the same 25-run sweep, "
                 "retrained where nothing is learnable",
                 x=0.004, ha="left", y=1.19, fontsize=11, fontweight="bold", color=INK)
    fig.text(0.004, 1.085, "The leak clears the floor on pure noise; the leak-free arms do "
             "not clear it on the real market — and their selection statistic is higher on noise.",
             fontsize=8.5, color=INK2, ha="left", va="top")
    note(fig, "Source: hrt/artifacts/null_sweep.json, runs_null/ and passive_synth_s*.json via "
              "analysis/null_sweep.py (sweep: hrt/slurm/05_sweep_null.sbatch). 25 null runs on 4 "
              "martingale-difference panels against 60 real runs, identical 501,760-step budget; "
              "each run is differenced against a do-nothing floor measured in the same environment "
              "on the same panel.")
    save(fig, "fig11_null_sweep.png")


FIGURES = [fig_falsification, fig_dsr, fig_cost_rerank, fig_censoring, fig_pooling,
           fig_persistence, fig_imbalance, fig_spread, fig_crypto, fig_text, fig_null_sweep]

if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    only = sys.argv[1:]
    for f in FIGURES:
        if only and not any(o in f.__name__ for o in only):
            continue
        print(f"{f.__name__} ...")
        f()
