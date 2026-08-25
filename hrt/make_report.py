#!/usr/bin/env python3
"""Render the reproduction report to a self-contained HTML artifact."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
sys.path.insert(0, HERE)
from report import PAPER, LABEL, PAPER_KEY, load_runs, agg

OUT = os.path.join(ART, "hrt_reproduction.html")

# ---------------------------------------------------------------- helpers
def sig(x, d=4, pct=False):
    if x is None or (isinstance(x, float) and x != x):
        return "&mdash;"
    return f"{x*100:+.2f}%" if pct else f"{x:+.{d}f}"


def svg_lines(series, width=760, height=300, pad=(46, 14, 30, 52), colors=None,
              ylabel="growth of $1"):
    """Minimal multi-line chart; series = {name: [values]} rebased to 1.0."""
    l, r, b, t = pad[3], pad[1], pad[2], pad[0]
    W, H = width, height
    xs_max = max(len(v) for v in series.values())
    lo = min(min(v) for v in series.values())
    hi = max(max(v) for v in series.values())
    span = hi - lo or 1.0
    lo, hi = lo - span * 0.06, hi + span * 0.06
    X = lambda i, n: l + (W - l - r) * (i / max(n - 1, 1))
    Y = lambda v: t + (H - t - b) * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{ylabel}" '
             f'preserveAspectRatio="xMidYMid meet">']
    # gridlines
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = Y(v)
        parts.append(f'<line class="grid" x1="{l}" y1="{y:.1f}" x2="{W-r}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{l-8}" y="{y+3.5:.1f}" text-anchor="end">{v:.2f}</text>')
    parts.append(f'<line class="axis" x1="{l}" y1="{Y(1.0):.1f}" x2="{W-r}" y2="{Y(1.0):.1f}"/>')

    for i, (name, vals) in enumerate(series.items()):
        c = colors[i % len(colors)]
        n = len(vals)
        d = " ".join(("M" if j == 0 else "L") + f"{X(j,n):.1f},{Y(v):.1f}"
                     for j, v in enumerate(vals))
        dash = ' stroke-dasharray="5 3"' if name.startswith("S&P") or "Passive" in name else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.9"{dash} '
                     f'stroke-linejoin="round"/>')
    parts.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:{colors[i%len(colors)]}"></i>{n}</span>'
        for i, n in enumerate(series))
    return f'<div class="chart">{"".join(parts)}</div><div class="legend">{legend}</div>'


def build():
    runs = load_runs()
    base = json.load(open(os.path.join(ART, "baselines.json")))
    passive = json.load(open(os.path.join(ART, "passive.json")))
    sigstrat = json.load(open(os.path.join(ART, "signal_strategy.json")))
    meta = json.load(open(os.path.join(ART, "universe.json")))
    fr_c = np.load(os.path.join(ART, "fr_causal.npz"), allow_pickle=True)
    fr_p = np.load(os.path.join(ART, "fr_paper.npz"), allow_pickle=True)

    ARMS = [
        ("hrt_causal", "HRT-FR", "signal timed causally; &alpha; decays per env step, as literally written"),
        ("hrt_causal_ep", "HRT-FR", "signal timed causally; &alpha; decays per episode"),
        ("hrt_paper", "HRT-FR", "the paper's stated timing; &alpha; decays per env step"),
        ("hrt_paper_ep", "HRT-FR", "the paper's stated timing; &alpha; decays per episode &mdash; the leak fully exploitable"),
        ("ppo_causal", "PPO", "standalone, no hierarchy"),
        ("ddpg_causal", "DDPG", "standalone, no hierarchy"),
    ]

    rows = {}
    for period in ("test2021", "test2022"):
        rr = []
        for arm, pk, note in ARMS:
            if arm not in runs:
                continue
            a = agg(runs[arm], period)
            rr.append(dict(arm=arm, label=LABEL.get(arm, arm), note=note,
                           n=a["n_seeds"], paper=PAPER[period].get(pk),
                           leak=arm.startswith("hrt_paper"),
                           cum=a["cum"], ann=a["ann"], vol=a["vol"],
                           sharpe=a["sharpe"], mdd=a["mdd"], turn=a["turnover"]))
        rows[period] = rr
    return dict(runs=runs, base=base, passive=passive, sigstrat=sigstrat, meta=meta,
                rows=rows,
                ic=dict(causal=float(fr_c["val_ic"]), paper=float(fr_p["val_ic"])))


CSS = """
:root{
  --paper:#eef1f3; --panel:#ffffff; --ink:#111820; --ink-soft:#4a5a66;
  --rule:#c9d2d8; --rule-soft:#dde4e8;
  --accent:#1b4d7a; --accent-soft:#e2ebf2;
  --alarm:#9c2b2b; --alarm-soft:#f6e7e5;
  --pass:#2f6b4f; --warn:#8a6318;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0e1418; --panel:#151d23; --ink:#e6edf1; --ink-soft:#9bb0bd;
    --rule:#2b3941; --rule-soft:#212d34;
    --accent:#7fb6e0; --accent-soft:#16242f;
    --alarm:#e08a84; --alarm-soft:#2a1a1a;
    --pass:#84c9a3; --warn:#d6ab5e;
  }
}
:root[data-theme="dark"]{
  --paper:#0e1418; --panel:#151d23; --ink:#e6edf1; --ink-soft:#9bb0bd;
  --rule:#2b3941; --rule-soft:#212d34;
  --accent:#7fb6e0; --accent-soft:#16242f;
  --alarm:#e08a84; --alarm-soft:#2a1a1a;
  --pass:#84c9a3; --warn:#d6ab5e;
}
*{box-sizing:border-box}
body{
  background:var(--paper); color:var(--ink); margin:0;
  font-family:"Source Serif 4",Georgia,serif; font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1080px;margin:0 auto;padding:0 28px 96px}
.col{max-width:68ch}
h1,h2,h3,.eyebrow,th,.label{font-family:"IBM Plex Sans Condensed",system-ui,sans-serif}
h1{font-size:clamp(2.1rem,5vw,3.3rem);line-height:1.04;font-weight:600;letter-spacing:-.015em;
   text-wrap:balance;margin:0 0 .5rem}
h2{font-size:1.55rem;font-weight:600;letter-spacing:-.01em;text-wrap:balance;margin:0}
h3{font-size:1.06rem;font-weight:600;margin:0 0 .35rem;letter-spacing:.005em}
p{margin:0 0 1.05rem}
a{color:var(--accent)}
code,.mono,td.num,th.num{font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums}
code{font-size:.86em;background:var(--accent-soft);padding:.1em .36em;border-radius:2px}

header.masthead{border-bottom:1px solid var(--rule);padding:64px 0 34px;margin-bottom:44px}
.eyebrow{font-size:.72rem;text-transform:uppercase;letter-spacing:.16em;
  color:var(--accent);font-weight:600;margin-bottom:1.1rem}
.standfirst{font-size:1.16rem;color:var(--ink-soft);max-width:60ch;margin:0}
.byline{margin-top:1.6rem;font-family:"IBM Plex Mono",monospace;font-size:.76rem;
  color:var(--ink-soft);display:flex;flex-wrap:wrap;gap:.5rem 1.4rem}

section{display:flex;flex-direction:column;gap:1.1rem;margin:0 0 60px}
.sec-head{display:flex;align-items:baseline;gap:.85rem;border-top:1px solid var(--rule);
  padding-top:.85rem}
.sec-num{font-family:"IBM Plex Mono",monospace;font-size:.76rem;color:var(--ink-soft)}

.verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1px;
  background:var(--rule-soft);border:1px solid var(--rule-soft)}
.verdict{background:var(--panel);padding:20px 22px;display:flex;flex-direction:column;gap:.4rem}
.verdict .label{font-size:.7rem;text-transform:uppercase;letter-spacing:.13em;font-weight:600}
.verdict.no .label{color:var(--alarm)} .verdict.yes .label{color:var(--pass)}
.verdict.part .label{color:var(--warn)}
.verdict p{margin:0;font-size:.92rem;color:var(--ink-soft);line-height:1.5}
.verdict strong{color:var(--ink);font-weight:600}

.callout{border:1px solid var(--rule);border-top:3px solid var(--alarm);
  background:var(--panel);padding:22px 24px;display:flex;flex-direction:column;gap:.7rem}
.callout .label{font-size:.7rem;text-transform:uppercase;letter-spacing:.13em;
  font-weight:600;color:var(--alarm)}
.callout p{margin:0}

.tablewrap{overflow-x:auto;border:1px solid var(--rule);background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.86rem}
caption{text-align:left;padding:14px 16px 12px;font-family:"IBM Plex Sans Condensed",sans-serif;
  font-weight:600;font-size:.95rem;border-bottom:1px solid var(--rule-soft)}
caption span{font-family:"Source Serif 4",serif;font-weight:400;color:var(--ink-soft);
  display:block;font-size:.84rem;margin-top:.25rem}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--rule-soft);
  white-space:nowrap}
th{font-size:.7rem;text-transform:uppercase;letter-spacing:.09em;color:var(--ink-soft);
  font-weight:600}
th.num,td.num{text-align:right}
tbody tr:last-child td{border-bottom:none}
tr.rule-top td{border-top:1px solid var(--rule)}
td .sub{display:block;font-family:"Source Serif 4",serif;font-size:.78rem;
  color:var(--ink-soft);white-space:normal;max-width:30ch}
.paper-col{color:var(--ink-soft)}
.miss{color:var(--alarm);font-weight:600}
.hit{color:var(--pass);font-weight:600}
tr.leak td{background:var(--alarm-soft)}
tr.published td{background:var(--accent-soft);border-top:1px solid var(--accent);
  border-bottom:1px solid var(--accent)}

.chart{overflow-x:auto;border:1px solid var(--rule);background:var(--panel);padding:10px}
.chart svg{width:100%;height:auto;min-width:520px;display:block}
.grid{stroke:var(--rule-soft);stroke-width:1}
.axis{stroke:var(--rule);stroke-width:1;stroke-dasharray:2 2}
.tick{fill:var(--ink-soft);font-size:10px;font-family:"IBM Plex Mono",monospace}
.legend{display:flex;flex-wrap:wrap;gap:.45rem 1.2rem;margin-top:.6rem;
  font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--ink-soft)}
.key{display:inline-flex;align-items:center;gap:.42rem}
.key i{width:16px;height:3px;display:inline-block;border-radius:1px}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:26px}

ol.steps{list-style:none;counter-reset:s;margin:0;padding:0;
  display:flex;flex-direction:column;gap:1px;background:var(--rule-soft);
  border:1px solid var(--rule-soft)}
ol.steps li{counter-increment:s;background:var(--panel);padding:16px 20px 16px 60px;
  position:relative}
ol.steps li::before{content:counter(s,decimal-leading-zero);position:absolute;left:20px;top:17px;
  font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--accent);font-weight:500}
ol.steps p{margin:0;font-size:.92rem;color:var(--ink-soft)}
ol.steps strong{color:var(--ink);font-family:"IBM Plex Sans Condensed",sans-serif;
  font-weight:600;font-size:.98rem;display:block;margin-bottom:.15rem}

dl.needs{margin:0;display:flex;flex-direction:column;gap:1px;background:var(--rule-soft);
  border:1px solid var(--rule-soft)}
dl.needs>div{background:var(--panel);padding:18px 22px}
dl.needs dt{font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:600;
  margin-bottom:.3rem;display:flex;gap:.7rem;align-items:baseline;flex-wrap:wrap}
dl.needs dd{margin:0;font-size:.92rem;color:var(--ink-soft)}
.chip{font-family:"IBM Plex Mono",monospace;font-size:.66rem;text-transform:uppercase;
  letter-spacing:.08em;padding:.15em .5em;border:1px solid var(--rule);
  color:var(--ink-soft);border-radius:2px;font-weight:400}
.chip.block{color:var(--alarm);border-color:var(--alarm)}
.chip.closed{color:var(--pass);border-color:var(--pass)}
.chip.dead{color:var(--alarm);border-color:var(--alarm)}

footer{border-top:1px solid var(--rule);padding-top:20px;color:var(--ink-soft);font-size:.82rem}
@media (prefers-reduced-motion:no-preference){
  a{transition:color .15s ease}
}
@media (max-width:640px){
  body{font-size:16px} .wrap{padding:0 18px 64px} header.masthead{padding-top:40px}
}
"""


def render(D):
    R = []
    A = R.append
    A('<title>Reproducing HRT</title>')
    A('<link rel="preconnect" href="https://fonts.googleapis.com">')
    A('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@400;600&'
      'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">')
    A(f"<style>{CSS}</style>")
    A('<div class="wrap">')
    return R


C_LINE = ["#1b4d7a", "#9c2b2b", "#2f6b4f", "#8a6318", "#6b4c8a", "#3d7f8a"]


def metric_table(D, period, year_label):
    rows = D["rows"][period]
    base, passive = D["base"][period], D["passive"][period]
    h = [f'<div class="tablewrap"><table><caption>{year_label}'
         f'<span>Reproduction (mean &plusmn; s.d. over seeds) against the published '
         f'Table&nbsp;1 figure for the same strategy. Sharpe = annualised return / '
         f'annualised volatility, r<sub>f</sub>&nbsp;=&nbsp;0, matching the paper.</span>'
         f'</caption><thead><tr>'
         '<th>strategy</th><th class="num">cum ret</th><th class="num">ann ret</th>'
         '<th class="num">ann vol</th><th class="num">Sharpe</th><th class="num">max DD</th>'
         '<th class="num">turnover</th><th class="num paper-col">paper cum</th>'
         '<th class="num paper-col">paper Sharpe</th></tr></thead><tbody>']

    for r in rows:
        leak = ' class="leak"' if r.get("leak") else ""
        pc = f'{r["paper"]["cum"]:+.4f}' if r["paper"] else "&mdash;"
        ps = f'{r["paper"]["sharpe"]:+.4f}' if r["paper"] else "&mdash;"
        sd = lambda t: f'<span style="opacity:.5">&plusmn;{t[1]:.3f}</span>' if r["n"] > 1 else ""
        h.append(
            f'<tr{leak}><td>{r["label"]} <span class="sub">{r["note"]} '
            f'&middot; n={r["n"]} seeds</span></td>'
            f'<td class="num">{r["cum"][0]:+.4f} {sd(r["cum"])}</td>'
            f'<td class="num">{r["ann"][0]:+.4f}</td>'
            f'<td class="num">{r["vol"][0]:.4f}</td>'
            f'<td class="num">{r["sharpe"][0]:+.4f} {sd(r["sharpe"])}</td>'
            f'<td class="num">{r["mdd"][0]:+.4f}</td>'
            f'<td class="num">{r["turn"][0]:.4f}</td>'
            f'<td class="num paper-col">{pc}</td>'
            f'<td class="num paper-col">{ps}</td></tr>')

    passives = [
        ("Passive buy &amp; hold", "deploy once inside the same env, then sit still", passive, None),
        ("Equal-weight buy &amp; hold", "weight-space, no trading frictions after entry",
         base["equal_weight_bh"], None),
        ("Minimum variance", "re-optimised daily, long-only, net of cost",
         base["min_variance_net"], PAPER[period]["Min-Var"]),
        ("S&amp;P 500 (^GSPC)", "index price return", base["sp500"], PAPER[period]["S&P500"]),
    ]
    for i, (name, note, m, pap) in enumerate(passives):
        cls = ' class="rule-top"' if i == 0 else ""
        pc = f'{pap["cum"]:+.4f}' if pap else "&mdash;"
        ps = f'{pap["sharpe"]:+.4f}' if pap else "&mdash;"
        turn = f'{m["turnover"]:.4f}' if "turnover" in m else "&mdash;"
        h.append(
            f'<tr{cls}><td>{name} <span class="sub">{note}</span></td>'
            f'<td class="num">{m["cum_return"]:+.4f}</td>'
            f'<td class="num">{m["ann_return"]:+.4f}</td>'
            f'<td class="num">{m["ann_vol"]:.4f}</td>'
            f'<td class="num">{m["sharpe"]:+.4f}</td>'
            f'<td class="num">{m["max_drawdown"]:+.4f}</td>'
            f'<td class="num">{turn}</td>'
            f'<td class="num paper-col">{pc}</td>'
            f'<td class="num paper-col">{ps}</td></tr>')
    h.append("</tbody></table></div>")
    return "".join(h)


def curves_figure(D, period, title):
    runs, base, passive = D["runs"], D["base"][period], D["passive"][period]
    series = {}
    for arm in ("hrt_causal", "ddpg_causal", "ppo_causal"):
        if arm in runs:
            V = np.array([r[period]["values"] for r in runs[arm]], float)
            series[LABEL[arm]] = list((V / V[:, :1]).mean(0))
    series["Passive buy & hold"] = [v / passive["values"][0] for v in passive["values"]]
    sp = np.array(base["series"]["sp500"], float)
    series["S&P 500"] = list(sp / sp[0])
    return (f'<div><h3>{title}</h3>{svg_lines(series, colors=C_LINE)}</div>')


PIPELINE = [
    ("Point-in-time universe",
     "The paper says &ldquo;S&amp;P&nbsp;500&rdquo; but never says <em>which</em> 500. "
     "Wikipedia's index change log was parsed and rewound from today's membership to "
     "1&nbsp;January&nbsp;2015, recovering 503 constituents. Only 370 have a complete "
     "2014&ndash;2022 price history in the local mirror."),
    ("Alpha158 features",
     "All 158 Qlib Alpha158 features (9 k-bar, 4 price, 29 rolling &times; 5 windows) "
     "reimplemented from the handler spec over a 2414&nbsp;&times;&nbsp;370 panel. "
     "17 representative features across every family were checked against an independent "
     "pandas implementation and match to float32 precision."),
    ("Forward-return forecaster",
     "Encoder-only Transformer, 10-day lookback, linear head, trained on 2015&ndash;2019 "
     "and selected on 2020 validation IC &mdash; the HLC's signal source. A ridge on the "
     "same panel was fitted as a control to separate signal quality from architecture."),
    ("Trading environment",
     "FinRL-style multi-stock env after Liu et al. (2018): state <code>[p, h, b]</code>, "
     "integer share lots capped at h<sub>max</sub>&nbsp;=&nbsp;100, $1M initial capital, "
     "0.1% proportional cost, execution at the open. Cash-sequenced fills in randomised "
     "order so no name is starved by its index position."),
    ("Bi-level controllers",
     "PPO high-level controller over a factorised 3<sup>N</sup> direction space, DDPG "
     "low-level controller sizing each trade. Both written once and shared with the "
     "standalone baselines, so the measured gap is the hierarchy and not two different "
     "libraries."),
    ("Phased alternating training",
     "Each iteration collects a rollout, blends the HLC reward as "
     "&alpha;<sub>t</sub>&middot;alignment&nbsp;+&nbsp;(1&minus;&alpha;<sub>t</sub>)&middot;LLC "
     "return with &alpha;<sub>t</sub>&nbsp;=&nbsp;e<sup>&minus;0.001t</sup>, then updates "
     "both controllers. 5&times;10<sup>5</sup> steps, checkpoint selected on 2020 Sharpe, "
     "evaluated once on 2021 and once on 2022."),
]

NEEDS = [
    ("News text and sentiment scores", "dead end",
     "The full HRT scores each stock daily from 10 sampled news items through "
     "<code>fingpt-sentiment_llama2-13b_lora</code>. Everything reported here is therefore "
     "HRT-FR, the paper's own no-sentiment ablation. This has now been probed with a live "
     "token rather than assumed: <code>/news/{symbol}</code> caps at 200 items, "
     "<strong>ignores every date parameter</strong> "
     "(<code>from</code>/<code>to</code>, <code>start</code>/<code>end</code>, "
     "<code>published_after</code>) and ignores paging, serving only the trailing fortnight; "
     "<code>since</code>/<code>until</code> returns HTTP&nbsp;500. The 2015&ndash;2019 "
     "training window is unreachable, so this channel cannot be built from this source at "
     "any scale of hardware. The workable route is the dataset HRT's own v2 revision uses "
     "&mdash; FNSPID, via the FinRL-DeepSeek benchmark, which ships precomputed LLM sentiment "
     "and risk scores &mdash; but that moves the universe from the S&amp;P 500 to an 89-name "
     "Nasdaq set, making it a different experiment rather than a port."),
    ("Prices for delisted securities", "blocking",
     "133 of the 503 point-in-time 2015 constituents have no usable history here, and "
     "<strong>108 of those 133 are names the index itself removed</strong> between 2015 and "
     "2022 &mdash; ACE, ALTR, ALXN, APC, BBBY, BRCM, CA and the rest of the acquired-or-failed "
     "cohort. The surviving 370 are not a random sample. CRSP, Sharadar or Norgate would "
     "close the gap; yfinance, the paper's stated source, cannot retrieve most of these "
     "tickers at all."),
    ("Price-only OHLCV", "closed",
     "findata's close is split <em>and</em> dividend adjusted while the paper's Yahoo "
     "convention is unstated, so a price-only panel was reconstructed from the "
     "<code>dividends</code> table by unwinding the cumulative adjustment factor back from "
     "the vendor's own 2026 reference bar. It validates to within 0.5% of known raw opens "
     "for names without spin-offs (AAPL &minus;0.0%, MSFT &minus;0.0%, XOM &minus;0.1%, "
     "JNJ &minus;0.5%, KO &minus;0.2%); IBM and AT&amp;T are off by 4% and 20% because "
     "Kyndryl and Warner Bros. Discovery were spin-offs, not cash dividends, and a "
     "dividend-only unwind cannot recover them. The measured tailwind is "
     "<strong>2.63 points in 2021 and 2.26 in 2022</strong> &mdash; enough to matter against "
     "a headline 2022 return of +2.29%, but not enough to change any conclusion here."),
    ("The authors' universe and label code", "unavailable",
     "No implementation was ever released. Point-in-time versus current-list membership, "
     "the handling of names that delisted mid-sample, and the exact label expression all "
     "have to be inferred from prose. The reproduction pins each choice explicitly and "
     "reports both readings where the prose is ambiguous."),
]

DEVIATIONS = [
    ("370 names, not 500", "Only 370 of the reconstructed 2015 index have complete price "
     "history locally. Smaller cross-section, same mechanics."),
    ("No sentiment channel", "HRT-FR is reproduced; full HRT is not, for want of news data."),
    ("Own PPO / DDPG", "The paper used FinRL's Stable-Baselines3 agents. Writing both once "
     "and sharing them with the baselines isolates the hierarchy, but means absolute levels "
     "are not directly comparable to a FinRL run &mdash; and required the factorisation fixes "
     "in section 03, without which the high-level policy never learns at all."),
    ("4 seeds, not 10", "The paper averages 10 seeds. Seed dispersion here is wide "
     "(HRT-FR 2022 spans roughly 20 points across seeds), so the means below are indicative "
     "of direction and magnitude rather than precisely estimated."),
    ("Alignment reward averaged", "The paper sums the per-stock alignment reward over N "
     "stocks, which at N=370 dwarfs the scaled portfolio-value term by two orders of "
     "magnitude. It is averaged here so both terms in the HLC reward are on the same scale."),
    ("Larger, less frequent DDPG updates", "One gradient step of 1024 samples every 4 env "
     "steps rather than 256 every step &mdash; identical samples per env step, ~4&times; "
     "fewer kernel launches on a launch-bound GPU."),
    ("Total-return prices", "The local mirror's OHLCV is dividend adjusted; the paper's "
     "Yahoo convention is unstated."),
    ("VWAP proxy", "No vendor VWAP, so the Alpha158 VWAP0 feature uses (H+L+C)/3. The paper "
     "also derives VWAP from OHLCV without saying how."),
]


DIAG_ROWS = [
    ("Stock PPO &mdash; entropy and ratio summed over all N dimensions",
     "+0.000", "0.333", "1.098", "no"),
    ("Per-dimension objective, entropy averaged over dimensions",
     "+0.028", "0.331", "0.958", "partly"),
    ("&hellip; plus per-stock credit from the paper's own reward decomposition",
     "+0.610", "0.028", "0.344", "yes"),
]

CHIP = {"blocking": "block", "dead end": "dead", "closed": "closed"}

def section(num, title, body):
    return (f'<section><div class="sec-head"><span class="sec-num">{num}</span>'
            f'<h2>{title}</h2></div>{body}</section>')


def activity_table(D):
    """Turnover, breadth and cost drag -- the paper's inertia and diversification
    claims, and the reason its net result is so cost-sensitive."""
    runs = D["runs"]
    order = [("ppo_causal", "PPO (standalone)"), ("ddpg_causal", "DDPG (standalone)"),
             ("hrt_causal", "HRT-FR causal, &alpha;/step"),
             ("hrt_causal_ep", "HRT-FR causal, &alpha;/episode"),
             ("hrt_paper_ep", "HRT-FR paper timing, &alpha;/episode")]
    h = ['<div class="tablewrap"><table><caption>Trading behaviour in 2022'
         '<span>Cost drag is the transaction cost actually paid over the year, as a fraction '
         'of starting capital. &ldquo;Gross&rdquo; adds it back to the net return, so the '
         'last two columns separate what the policy earned from what the turnover cost it. '
         'Passive floor: &minus;5.64%.</span></caption><thead><tr><th>agent</th>'
         '<th class="num">daily turnover</th><th class="num">names traded/day</th>'
         '<th class="num">names ever traded</th><th class="num">trade HHI</th>'
         '<th class="num">cost drag</th><th class="num">net</th><th class="num">gross</th>'
         '</tr></thead><tbody>']
    for arm, name in order:
        if arm not in runs:
            continue
        rs = runs[arm]
        g = lambda k: np.mean([r["test2022"][k] for r in rs if k in r["test2022"]]) \
            if any(k in r["test2022"] for r in rs) else float("nan")
        turn, cost = g("turnover"), g("cost_paid") / 1e6
        net = np.mean([r["test2022"]["cum_return"] for r in rs])
        npd, ever, hhi = g("active_names_per_day"), g("names_ever_traded"), g("trade_concentration_hhi")
        f = lambda x, d=1: "&mdash;" if x != x else f"{x:,.{d}f}"
        cls = ' class="leak"' if arm.startswith("hrt_paper") else ""
        h.append(f'<tr{cls}><td>{name}</td><td class="num">{turn:.4f}</td>'
                 f'<td class="num">{f(npd)}</td><td class="num">{f(ever,0)}</td>'
                 f'<td class="num">{"&mdash;" if hhi != hhi else f"{hhi:.5f}"}</td>'
                 f'<td class="num">{cost*100:.2f}%</td>'
                 f'<td class="num">{net*100:+.2f}%</td>'
                 f'<td class="num">{(net+cost)*100:+.2f}%</td></tr>')
    h.append('</tbody></table></div>')
    return "".join(h)



def bracket_table(D):
    """The single clearest exhibit: published values sit between the causal and
    leaked implementations, matching neither."""
    r21 = {r["arm"]: r for r in D["rows"]["test2021"]}
    r22 = {r["arm"]: r for r in D["rows"]["test2022"]}
    order = [("hrt_causal", "Causal timing, &alpha; per step"),
             ("hrt_causal_ep", "Causal timing, &alpha; per episode"),
             (None, "Published HRT-FR"),
             ("hrt_paper", "Paper's timing, &alpha; per step"),
             ("hrt_paper_ep", "Paper's timing, &alpha; per episode")]
    h = ['<div class="tablewrap"><table><caption>The published result is bracketed, not '
         'matched<span>Cumulative return by test year. Everything above the published row '
         'uses a leak-free signal; everything below exploits the timing the paper specifies. '
         'Neither end lands on the published figures.</span></caption><thead><tr>'
         '<th>implementation</th><th class="num">2021</th><th class="num">2022</th>'
         '<th class="num">2021 Sharpe</th><th class="num">2022 Sharpe</th>'
         '<th class="num">seeds</th></tr></thead><tbody>']
    for arm, name in order:
        if arm is None:
            p1, p2 = PAPER["test2021"]["HRT-FR"], PAPER["test2022"]["HRT-FR"]
            h.append(f'<tr class="published"><td><strong>{name}</strong></td>'
                     f'<td class="num"><strong>{p1["cum"]:+.4f}</strong></td>'
                     f'<td class="num"><strong>{p2["cum"]:+.4f}</strong></td>'
                     f'<td class="num">{p1["sharpe"]:+.2f}</td>'
                     f'<td class="num">{p2["sharpe"]:+.2f}</td>'
                     f'<td class="num">10</td></tr>')
            continue
        if arm not in r21:
            continue
        a, b = r21[arm], r22[arm]
        cls = ' class="leak"' if arm.startswith("hrt_paper") else ""
        h.append(f'<tr{cls}><td>{name}</td>'
                 f'<td class="num">{a["cum"][0]:+.4f}</td>'
                 f'<td class="num">{b["cum"][0]:+.4f}</td>'
                 f'<td class="num">{a["sharpe"][0]:+.2f}</td>'
                 f'<td class="num">{b["sharpe"][0]:+.2f}</td>'
                 f'<td class="num">{a["n"]}</td></tr>')
    h.append('</tbody></table></div>')
    return "".join(h)



def verdicts(D):
    """Headline calls, computed from the numbers rather than asserted."""
    v = []
    r21 = {r["arm"]: r for r in D["rows"]["test2021"]}
    r22 = {r["arm"]: r for r in D["rows"]["test2022"]}
    p21, p22 = D["passive"]["test2021"], D["passive"]["test2022"]

    best_causal = None
    for k in ("hrt_causal_ep", "hrt_causal"):
        if k in r22:
            best_causal = k
            break
    if best_causal:
        c22, c21 = r22[best_causal], r21[best_causal]
        v.append(("no", "Headline claim",
                  f'Under causal timing HRT-FR returns '
                  f'<strong>{c22["cum"][0]*100:+.2f}%</strong> in 2022 against a published '
                  f'<strong>+2.29%</strong>, and {c21["cum"][0]*100:+.2f}% in 2021 against '
                  f'<strong>+39.83%</strong>. Neither year reproduces.'))
    if "hrt_paper_ep" in r22:
        l21, l22 = r21["hrt_paper_ep"], r22["hrt_paper_ep"]
        v.append(("no", "But the leak overshoots",
                  f'Wired to the timing the paper specifies, the same agent returns '
                  f'<strong>{l21["cum"][0]*100:+.1f}%</strong> and '
                  f'<strong>{l22["cum"][0]*100:+.1f}%</strong> &mdash; far above the published '
                  f'figures, not equal to them. The published numbers sit between the two arms.'))
    v.append(("part", "Passive floor",
              f'Deploying the cash once and sitting still, inside the same environment, returns '
              f'{p21["cum_return"]*100:+.2f}% and {p22["cum_return"]*100:+.2f}%. Most agents here '
              f'do not clear it in the bear year.'))
    v.append(("yes", "Motivation",
              'The pathology the paper is built to fix is real. Standalone DDPG collapses to '
              'buy-and-hold &mdash; its whole annual turnover is the day-one deployment &mdash; '
              'while the hierarchy trades throughout.'))
    return v


def main():
    D = build()
    R = render(D)
    A = R.append
    r21 = {r["arm"]: r for r in D["rows"]["test2021"]}
    r22 = {r["arm"]: r for r in D["rows"]["test2022"]}
    nseed = max([r["n"] for r in D["rows"]["test2021"]] or [0])

    A('<header class="masthead"><div class="col">')
    A('<div class="eyebrow">Reproduction study &middot; arXiv:2410.14927v1</div>')
    A('<h1>Reproducing the Hierarchical Reinforced Trader</h1>')
    A('<p class="standfirst">A bi-level PPO&ndash;DDPG trader reports beating the '
      'S&amp;P&nbsp;500 in both a bull and a bear year. Rebuilt from the paper text on '
      'point-in-time index data, it lands well short of the published figures when its signal '
      'is timed honestly, and well past them when the signal is timed as the paper specifies. '
      'The published result sits in between &mdash; in an interval that requires the '
      'look-ahead to be there.</p>')
    A(f'<div class="byline"><span>370 point-in-time S&amp;P 500 names</span>'
      f'<span>2015&ndash;19 train &middot; 2020 validate &middot; 2021&ndash;22 test</span>'
      f'<span>{nseed} seeds &times; 5&times;10<sup>5</sup> steps</span></div>')
    A('</div></header>')

    vs = verdicts(D)
    A(section("01", "What reproduced",
        '<div class="verdicts">' + "".join(
            f'<div class="verdict {k}"><span class="label">{t}</span><p>{b}</p></div>'
            for k, t, b in vs) + '</div>'))

    # ---- the leak
    ss = D["sigstrat"]
    leak_tbl = ['<div class="tablewrap"><table><caption>The same strategy under three signal '
                'timings<span>Top-30 equal-weight long book, rebalanced daily at the open, '
                'paying the same 0.1% cost the RL agents pay. Nothing here is reinforcement '
                'learning &mdash; it is the forecast alone.</span></caption>'
                '<thead><tr><th>signal timing</th><th class="num">2021 cum</th>'
                '<th class="num">2022 cum</th><th class="num">2021 Sharpe</th>'
                '<th class="num">test IC</th></tr></thead><tbody>']
    for mode, name, ic in (("causal", "Causal &mdash; features at close <em>t</em>, target open <em>t</em>+1&rarr;<em>t</em>+2", "+0.0118"),
                           ("paper", "As written &mdash; target open <em>t</em>&rarr;<em>t</em>+1", "+0.8155"),
                           ("shuffle", "Shuffled control", "&minus;0.004")):
        a, b = ss[f"{mode}_k30_test2021"], ss[f"{mode}_k30_test2022"]
        cls = ' class="leak"' if mode == "paper" else ""
        A_ = lambda m: (f'{m["cum_return"]*100:,.0f}%' if abs(m["cum_return"]) > 5
                        else f'{m["cum_return"]*100:+.2f}%')
        leak_tbl.append(f'<tr{cls}><td>{name}</td><td class="num">{A_(a)}</td>'
                        f'<td class="num">{A_(b)}</td>'
                        f'<td class="num">{a["sharpe"]:,.2f}</td>'
                        f'<td class="num">{ic}</td></tr>')
    leak_tbl.append('</tbody></table></div>')
    leak_html = "".join(leak_tbl)

    A(section("02", "The signal leaks before any agent is trained",
        '<div class="col">'
        '<p>The high-level controller is fed a predicted forward return, defined in the paper '
        'as the change in opening price <em>from day T to day T+1</em>, produced from Qlib '
        'Alpha158 features observed through day <em>T</em>. Day <em>T</em>&rsquo;s close sits '
        'inside that window. Alpha158 carries <code>KMID = (close&minus;open)/open</code> and '
        'twelve more features built on the same bar, so the first leg of the target is handed '
        'to the model as an input.</p>'
        '<p>Qlib&rsquo;s own Alpha158 handler labels with '
        '<code>Ref($close,-2)/Ref($close,-1)-1</code>, which skips precisely this window. The '
        'paper departs from that default without remarking on it.</p></div>'
        + leak_html +
        '<div class="col"><div class="callout"><span class="label">What this means</span>'
        '<p>A 1,089&times; return in a single year is not a trading result; it is a model '
        'reading tomorrow&rsquo;s bar. No implementation was released, so it cannot be '
        'confirmed that the published numbers were produced this way. What can be said is '
        'that the procedure <em>as described</em> is not implementable without leakage, and '
        'that every figure below uses the causal reading instead.</p></div></div>'))

    diag_html = ('<div class="tablewrap"><table><caption>Can the high-level controller '
        'learn to follow its own signal?<span>PPO trained on the directional-alignment '
        'reward alone, under the leaked signal, where <code>action = sign(forecast)</code> '
        'is close to free to learn. 30,720 steps. Maximum entropy for a 3-way choice is '
        'ln&nbsp;3 = 1.0986.</span></caption><thead><tr><th>implementation</th>'
        '<th class="num">mean alignment</th><th class="num">frac. hold</th>'
        '<th class="num">entropy/dim</th><th class="num">learns?</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td style="white-space:normal;max-width:44ch">{a}</td>'
            f'<td class="num">{b}</td><td class="num">{c}</td><td class="num">{d}</td>'
            f'<td class="num {"miss" if e=="no" else ("hit" if e=="yes" else "")}">{e}</td></tr>'
            for a, b, c, d, e in DIAG_ROWS)
        + '</tbody></table></div>')

    A(section("03", "PPO does not survive a 3<sup>N</sup> action space unchanged",
        '<div class="col">'
        '<p>The high-level controller picks one of three directions for each of 370 stocks, '
        'a factorised categorical over 3<sup>370</sup> joint actions. A stock PPO sums the '
        'entropy bonus and the importance ratio across action dimensions, which is harmless '
        'at the handful of dimensions such code is normally run on. At N&nbsp;=&nbsp;370 the '
        'entropy bonus is <code>0.01 &times; 370 &times; ln&nbsp;3 &asymp; 4.06</code> against '
        'a policy-gradient term of order 0.01. The optimiser simply maximises entropy and the '
        'policy never leaves uniform random.</p>'
        '<p>This is easy to miss, because a random high-level policy still produces '
        'plausible-looking returns &mdash; it deploys capital and tracks the market. It only '
        'shows up if you ask the controller to learn something it cannot fail at.</p></div>'
        + diag_html +
        '<div class="col"><p>Two corrections restore learning, and both follow the paper rather '
        'than departing from it. The clipped objective and entropy are taken per dimension and '
        'averaged. And credit is assigned per stock: the paper&rsquo;s HLC reward is '
        '<em>&alpha;&middot;&Sigma;<sub>i</sub> r<sub>i</sub><sup>align</sup> + '
        '(1&minus;&alpha;)&middot;r<sup>l</sup></em>, a sum in which each term depends only on '
        'that stock&rsquo;s own action &mdash; collapsing it to one scalar before the update '
        'throws away exactly the structure that makes 370 simultaneous decisions learnable.</p>'
        '<div class="callout"><span class="label">Why the &alpha; schedule now matters</span>'
        '<p>The paper sets &alpha;<sub>t</sub> = e<sup>&minus;0.001t</sup> without saying what '
        '<em>t</em> counts. Per env step, &alpha; is spent by step 3,000 of 500,000 &mdash; '
        'per-stock credit is switched off for 99.4% of training and the controller is left '
        'learning 370 decisions from a single scalar. Per episode, it decays across the whole '
        'run. Both readings are reported below; the difference is not cosmetic.</p></div>'
        '</div>'))

    ce = r21.get("hrt_causal_ep") or r21.get("hrt_causal")
    ce22 = r22.get("hrt_causal_ep") or r22.get("hrt_causal")
    c21, c22 = ce["cum"][0], ce22["cum"][0]
    ppo21, ddpg21 = r21["ppo_causal"]["cum"][0], r21["ddpg_causal"]["cum"][0]
    l21 = r21["hrt_paper"]["cum"][0]
    l22 = r22["hrt_paper"]["cum"][0]
    A(section("04", "What the reproduction actually shows",
        bracket_table(D) +
        '<div class="col">'
        f'<p>Read leak-free, HRT-FR reaches {c21*100:+.1f}% in 2021 and {c22*100:+.1f}% in '
        f'2022, against published figures of +39.8% and +2.3%. The paper&rsquo;s ordering does '
        f'not survive either: it reports HRT-FR&nbsp;&gt;&nbsp;DDPG&nbsp;&gt;&nbsp;PPO in both '
        f'years, where here the two flat agents finish ahead of the hierarchy in 2021 '
        f'({ppo21*100:+.1f}% and {ddpg21*100:+.1f}% against {c21*100:+.1f}%) and the three sit '
        f'within a point and a half of each other in 2022. Every one of them, the hierarchy '
        f'included, finishes below a portfolio that bought once and then did nothing.</p>'
        '<p>Wired to the timing the paper specifies, the same code overshoots the published '
        'numbers by a wide margin &mdash; and does so with unusually tight seed agreement, '
        'because an agent reading tomorrow&rsquo;s bar has little left to be uncertain about. '
        'So the published figures are not what full exploitation of the stated timing '
        'produces either.</p>'
        f'<p>What lies between the two ends is partial exploitation. The defective agent of '
        f'section&nbsp;03 is one way to arrive there: on the leaked signal but with crippled '
        f'credit assignment it returned +21.3% and &minus;9.7%, below the published values, '
        f'while the corrected agent on the same signal returns {l21*100:+.0f}% and '
        f'{l22*100:+.0f}%, above them. A weaker forecaster, a partially overlapping feature '
        f'window, or a less efficient learner would each land somewhere inside that interval. '
        f'None of this identifies what the authors ran &mdash; no code was released &mdash; '
        f'but the interval that reproduces their numbers is one that requires the '
        f'look-ahead to be present.</p></div>'))

    A(section("05", "Results against Table 1",
        metric_table(D, "test2021", "2021 &mdash; bullish market")
        + metric_table(D, "test2022", "2022 &mdash; bearish market")))

    _ce = D["runs"].get("hrt_causal_ep") or D["runs"]["hrt_causal"]
    drag = float(np.mean([r["test2022"]["cost_paid"] for r in _ce]) / 1e6)
    net22 = float(np.mean([r["test2022"]["cum_return"] for r in _ce]))
    turn = float(np.mean([r["test2022"]["turnover"] for r in _ce]))
    gross = net22 + drag
    floor = D["passive"]["test2022"]["cum_return"]
    A(section("06", "The edge is real and the turnover eats it",
        activity_table(D) +
        '<div class="col">'
        '<p>Two of the three pathologies the paper sets out to fix do measurably improve. '
        'Standalone PPO and DDPG barely trade at all &mdash; a daily turnover of 0.4% is '
        'entirely the day-one deployment of cash, and PPO touches 1.4 names a day and only '
        '165 of 370 names all year. The hierarchy trades 21&ndash;27 names a day, reaches '
        'essentially the whole universe, and halves the concentration of what it trades. '
        'Inertia and narrow diversification are real, and the bi-level structure genuinely '
        'addresses both.</p>'
        f'<p>But look at the last two columns. Gross of transaction costs, HRT-FR on an '
        f'honest signal returns {gross*100:+.2f}% in 2022 against a passive floor of '
        f'{floor*100:+.2f}% &mdash; {abs(gross-floor)*100:.1f} points of genuine edge. It then '
        f'pays {drag*100:.2f} points in commission to collect it, and finishes below the '
        f'floor. At {turn*100:.0f}% daily turnover the strategy churns its book roughly '
        f'{turn*252:.0f} times a year, and a flat 10&nbsp;bp cost assumption is doing an '
        f'enormous amount of load-bearing work.</p>'
        '<div class="callout" style="border-top-color:var(--accent)">'
        '<span class="label" style="color:var(--accent)">Where this points</span>'
        '<p>The interesting quantity in this architecture is not whether the hierarchy beats a '
        'flat agent &mdash; it does, gross &mdash; but whether it can be made to pay for '
        'itself. That makes turnover penalties and a realistic, non-constant cost model the '
        'load-bearing part of any follow-on work, not a detail to be bolted on afterwards. '
        'Notably, the paper\'s own v2 revision moves in exactly this direction, adding an '
        'explicit turnover budget and a turnover penalty to the low-level reward.</p></div>'
        '</div>'))

    A(section("07", "Equity curves",
        '<div class="figs">'
        + curves_figure(D, "test2021", "2021")
        + curves_figure(D, "test2022", "2022")
        + '</div>'))

    # ---- survivorship
    A(section("08", "The bear-year result is mostly the universe",
        '<div class="col">'
        '<p>The paper&rsquo;s most striking claim is 2022: HRT holds a small positive return '
        'while the S&amp;P&nbsp;500 loses 20%. But the benchmark and the trading universe are '
        'not the same population.</p>'
        '<p>Of the 503 constituents reconstructed as of January 2015, 133 have no usable price '
        'history in the data available here &mdash; and <strong>108 of those 133 are names the '
        'index itself removed</strong> during 2015&ndash;2022. What survives is the cohort that '
        'was never acquired and never failed. Holding all 370 of them in equal weight, with no '
        'model whatsoever, returns <strong>&minus;5.40%</strong> in 2022 against the '
        'index&rsquo;s &minus;19.91%. Fourteen points of apparent skill are available before '
        'anything is learned.</p>'
        '<p>This is not a defect peculiar to this reproduction. yfinance, the source the paper '
        'names, cannot retrieve most of those delisted tickers at all, so any &ldquo;S&amp;P 500&rdquo; '
        'universe assembled from it is filtered the same way &mdash; probably more severely, '
        'since the reconstruction here at least keeps 44 names the index dropped.</p>'
        '<p>The gap is not an artefact of the dividend convention either. Reconstructing a '
        'price-only panel from the dividend record puts the same equal-weight basket at '
        '<strong>&minus;7.66%</strong> for 2022 against the index&rsquo;s &minus;19.91% &mdash; '
        'still 12.3 points, like for like.</p></div>'))

    A(section("09", "How it was rebuilt",
        '<ol class="steps">' + "".join(
            f'<li><strong>{t}</strong><p>{b}</p></li>' for t, b in PIPELINE) + '</ol>'))

    A(section("10", "Data still needed",
        '<dl class="needs">' + "".join(
            f'<div><dt>{t}<span class="chip {"block" if s=="blocking" else ""}">{s}</span></dt>'
            f'<dd>{b}</dd></div>' for t, s, b in NEEDS) + '</dl>'))

    A(section("11", "Deviations from the paper",
        '<div class="tablewrap"><table><thead><tr><th>choice</th><th>reason</th></tr></thead>'
        '<tbody>' + "".join(
            f'<tr><td style="white-space:normal;max-width:22ch"><strong>{t}</strong></td>'
            f'<td style="white-space:normal">{b}</td></tr>' for t, b in DEVIATIONS)
        + '</tbody></table></div>'))

    A('<footer><div class="col">Zhao &amp; Welsch, &ldquo;Hierarchical Reinforced Trader (HRT): '
      'A Bi-Level Approach for Optimizing Stock Selection and Execution&rdquo;, '
      '<a href="https://arxiv.org/abs/2410.14927">arXiv:2410.14927v1</a>. '
      'Reproduction code in <code>~/fyp/hrt/</code>; every figure regenerable from '
      '<code>panel.npz</code> and the run JSONs.</div></footer>')
    A('</div>')

    open(OUT, "w").write("\n".join(R))
    print("wrote", OUT, f"{os.path.getsize(OUT)/1024:.0f} KB")


if __name__ == "__main__":
    main()
