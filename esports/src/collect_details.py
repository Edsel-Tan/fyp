"""Collect livestats `details` (wards, items, damage share) at the 60 s frame timestamps, one parquet per game.

usage: python collect_details.py OUTDIR [game_id_file]

For every frames60 row we request the details window starting at that timestamp and keep its first frame,
so rows align one-to-one with frames60 (column `ts60` is the frames60 timestamp). Cumulative per-player
fields: wardsPlaced, wardsDestroyed; point-in-time: items (as a comma string), control wards held
(item 2055), championDamageShare, killParticipation.
"""
import sys, os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import lolapi
from collect_frames import iso, pt

OUT = sys.argv[1] if __name__ == "__main__" else None
F60 = '../data/frames60'


def flat(f, ts60):
    r = {"ts60": ts60, "ts": f["rfc460Timestamp"]}
    for p in f["participants"]:
        i = p["participantId"]
        r[f"p{i}_wp"] = p.get("wardsPlaced")
        r[f"p{i}_wd"] = p.get("wardsDestroyed")
        items = p.get("items") or []
        r[f"p{i}_cw"] = sum(x == 2055 for x in items)
        r[f"p{i}_items"] = ",".join(map(str, items))
        r[f"p{i}_dmg"] = p.get("championDamageShare")
        r[f"p{i}_kp"] = p.get("killParticipation")
    return r


def one(gid):
    path = f"{OUT}/{gid}.parquet"
    if os.path.exists(path) or os.path.exists(path + ".none") or not os.path.exists(f"{F60}/{gid}.parquet"):
        return gid, "skip"
    ts = pd.read_parquet(f"{F60}/{gid}.parquet", columns=["ts"]).ts.tolist()
    rows, misses = [], 0
    for t in ts:
        d = lolapi.details(gid, iso(pt(t)))
        if not d or not d.get("frames"):
            misses += 1
            continue
        rows.append(flat(d["frames"][0], t))
    if not rows:
        open(path + ".none", "w").close()
        return gid, "none"
    df = pd.DataFrame(rows)
    df.insert(0, "game_id", gid)
    df.to_parquet(path)
    return gid, "ok" if misses == 0 else "partial"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ids = [l.strip() for l in open(sys.argv[2])]
    stats = {}
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 32))) as ex:
        futs = [ex.submit(one, g) for g in ids]
        for i, f in enumerate(as_completed(futs)):
            try:
                _, s = f.result()
            except Exception as e:
                s = "err:" + type(e).__name__
            stats[s] = stats.get(s, 0) + 1
            if i % 200 == 0:
                print(i, len(ids), stats, flush=True)
    print("DONE", stats, flush=True)
