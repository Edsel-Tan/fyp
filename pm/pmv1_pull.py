"""Download the cleaned trade layers of Polymarket-v1 [16] from Hugging Face.

The vendor tape in data/pm.sqlite covers 2026-05-21 -> 2026-09-04 and carries no
aggressor field, so RESULTS.md sec.3 rests on fifteen weeks of side-inferred
trades.  This dataset covers 2022-11-21 -> 2026-04-28 with the taker side taken
from blockchain settlement, and `neg_risk_market_id` gives the event grouping
pm/skill.py previously had to infer from market titles by token Jaccard.

Only the two cleaned trade layers are pulled (~16.8 GB):
  daily_aligned/        standard binary markets   (neg_risk=false)  13.2 GB
  daily_aligned_multi/  neg-risk multi-outcome    (neg_risk=true)    3.6 GB
OrderFilled/ (27 GB raw nominal tape, relayers not filtered) and CTF/ (8.5 GB
lifecycle logs) are not needed for the questions in RESULTS.md.

    python pm/pmv1_pull.py                     # both layers, resumable
    python pm/pmv1_pull.py --layer daily_aligned --workers 16
"""
import argparse, os, sys

REPO = "TimeSeventeen/Polymarket-v1"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pmv1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", action="append",
                    choices=["daily_aligned", "daily_aligned_multi", "CTF", "OrderFilled"])
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    layers = a.layer or ["daily_aligned", "daily_aligned_multi"]

    from huggingface_hub import snapshot_download
    os.makedirs(ROOT, exist_ok=True)
    p = snapshot_download(
        repo_id=REPO, repo_type="dataset", local_dir=ROOT,
        allow_patterns=[f"{L}/**" for L in layers] + ["README.md"],
        max_workers=a.workers,
    )
    for L in layers:
        d = os.path.join(ROOT, L)
        n = sum(len(f) for _, _, f in os.walk(d))
        b = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(d) for f in fs)
        print(f"{L}: {n} files, {b/1e9:.2f} GB", file=sys.stderr)
    print(p)


if __name__ == "__main__":
    main()
