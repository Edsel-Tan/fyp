"""Collect livestats frames for every completed game, one parquet per game.

usage: python collect_frames.py STEP_SECONDS OUTDIR [game_id_file]

Per game: the first window gives metadata (players, champions, patch) and the
start time; then one window per STEP seconds until the feed reports
`finished`. From each window we keep its first frame (plus the finished frame),
so a row is the game state at a known wall-clock instant. With STEP <= 10
the windows tile the game and every frame is kept (full feed resolution).
"""
import sys, os, json, datetime as dt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import lolapi

STEP = int(sys.argv[1]) if __name__ == "__main__" else 60
OUT = sys.argv[2] if __name__ == "__main__" else None
if OUT:
    os.makedirs(OUT, exist_ok=True)
MAXLEN = 100 * 60  # hard cap on game length (wall-clock seconds, incl. pauses)


def iso(t):
    t = t.replace(microsecond=0)
    t -= dt.timedelta(seconds=t.second % 10)
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def pt(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def flat(f):
    r = {"ts": f["rfc460Timestamp"], "state": f["gameState"]}
    for side in ("blue", "red"):
        t = f[f"{side}Team"]
        s = side[0]
        r[f"{s}_gold"] = t["totalGold"]
        r[f"{s}_kills"] = t["totalKills"]
        r[f"{s}_towers"] = t["towers"]
        r[f"{s}_inhibs"] = t["inhibitors"]
        r[f"{s}_barons"] = t["barons"]
        r[f"{s}_drakes"] = ",".join(t["dragons"])
        for p in t["participants"]:
            i = p["participantId"]
            for k in ("totalGold", "level", "kills", "deaths", "assists", "creepScore", "currentHealth", "maxHealth"):
                r[f"p{i}_{k}"] = p[k]
    return r


def one(gid):
    path = f"{OUT}/{gid}.parquet"
    if os.path.exists(path) or os.path.exists(path + ".none"):
        return gid, "skip"
    w = lolapi.window(gid)
    if not w or not w.get("frames"):
        open(path + ".none", "w").close()
        return gid, "none"
    meta = w["gameMetadata"]
    rows = [flat(w["frames"][0])]
    t0 = pt(w["frames"][0]["rfc460Timestamp"])
    t, misses, done = t0, 0, False
    while not done and (t - t0).total_seconds() < MAXLEN:
        t += dt.timedelta(seconds=STEP)
        w = lolapi.window(gid, iso(t))
        if not w or not w.get("frames"):
            misses += 1
            if misses >= 6:
                break
            continue
        misses = 0
        fr = w["frames"]
        if STEP <= 10:  # full resolution: keep every frame of the window
            rows += [flat(f) for f in fr]
        else:
            rows.append(flat(fr[0]))
        fin = [f for f in fr if f["gameState"] == "finished"]
        if fin:
            rows.append(flat(fin[0]))
            done = True
    df = pd.DataFrame(rows).drop_duplicates("ts")
    df.insert(0, "game_id", gid)
    df.attrs = {}
    df.to_parquet(path)
    with open(f"{OUT}/{gid}.meta.json", "w") as fh:
        json.dump(meta, fh)
    return gid, "finished" if done else "open"


if __name__ == "__main__":
    ids = [l.strip() for l in open(sys.argv[3])] if len(sys.argv) > 3 else \
        pd.read_parquet("../data/games.parquet").query("gstate=='completed'").game_id.tolist()
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
