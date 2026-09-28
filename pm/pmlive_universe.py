#!/usr/bin/env python3
"""Maintain the live Polymarket collection universe in data/pmlive.sqlite.

Policy is *sticky with a sports cap*: once an event enters the top-N by 24h
volume it is tracked until its markets resolve, so the panel has no survivorship
hole from re-ranking.  Sports events are capped because a single NFL game ships
~660 prop tokens -- 53 sports events carried 3,800 of 5,766 tokens at the same
24h volume as Politics' 744, so uncapped they crowd out everything else.

Two API traps are handled here and must not be "simplified" away:
  * gamma `limit` silently caps at 100.  Asking for 150 returns 100 with no
    error, so the top-N is paged by offset.
  * Cloudflare 403s python's default User-Agent on every polymarket host.
"""
import argparse, datetime as dt, json, os, sqlite3, time
import urllib.request, urllib.error, urllib.parse

GAMMA = "https://gamma-api.polymarket.com"
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, os.pardir, "data", "pmlive.sqlite")
UA = {"User-Agent": "fyp-pmlive/1.0"}
PAGE = 100  # gamma's real ceiling regardless of what you ask for

SPORTS = {"sports", "nfl", "nba", "mlb", "nhl", "soccer", "tennis", "golf", "ufc",
          "boxing", "cfb", "cbb", "epl", "laliga", "seriea", "bundesliga", "ligue1",
          "mls", "esports", "lol", "cs2", "dota", "valorant", "f1", "nascar", "cricket"}


def get(path, **kw):
    q = urllib.parse.urlencode(kw)
    for attempt in range(5):
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(f"{GAMMA}{path}?{q}", headers=UA), timeout=90)
            return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def get_raw(path):
    """GET a fully-formed path+query (repeated params gamma requires)."""
    for attempt in range(5):
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(f"{GAMMA}{path}", headers=UA), timeout=90)
            return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return []
            if e.code == 422:               # batch too large: caller must shrink
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS events(
        event_id TEXT PRIMARY KEY, slug TEXT, title TEXT, category TEXT,
        end_date TEXT, first_seen TEXT, last_seen TEXT,
        best_rank INT, max_vol24 REAL, capped INT DEFAULT 0)""")
    c.execute("""CREATE TABLE IF NOT EXISTS markets(
        condition_id TEXT PRIMARY KEY, event_id TEXT, question TEXT,
        group_title TEXT, end_date TEXT, active INT, closed INT,
        first_seen TEXT, last_seen TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS tokens(
        token_id TEXT PRIMARY KEY, tok INTEGER UNIQUE, condition_id TEXT,
        event_id TEXT, outcome TEXT, first_seen TEXT, last_seen TEXT,
        tracking INT DEFAULT 1)""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_tok_track ON tokens(tracking)")
    c.commit()
    return c


def category(e):
    for t in (e.get("tags") or []):
        lab = (t.get("label") or "").strip()
        if lab.lower().replace(" ", "") in SPORTS:
            return "Sports"
    for t in (e.get("tags") or []):
        lab = (t.get("label") or "").strip()
        if lab and lab.lower() not in ("all", "trending"):
            return lab
    return "none"


def live_markets(e):
    """Markets still quoting, richest first, with their token ids."""
    out = []
    for m in e.get("markets") or []:
        if not (m.get("active") and not m.get("closed")):
            continue
        try:
            ids = json.loads(m["clobTokenIds"])
            outs = json.loads(m.get("outcomes") or "[]")
        except Exception:
            continue
        if not ids:
            continue
        liq = m.get("liquidityNum") or m.get("liquidityClob") or 0
        out.append((float(liq or 0), m, ids, outs))
    out.sort(key=lambda r: -r[0])
    return out


def refresh(c, top_n, sports_cap, hard_cap):
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    evs, seen = [], set()
    for off in range(0, top_n, PAGE):
        b = get("/events", limit=PAGE, offset=off, closed="false", archived="false",
                active="true", order="volume24hr", ascending="false")
        if not b:
            break
        for e in b:
            if e["id"] not in seen:
                seen.add(e["id"]); evs.append(e)
        if len(b) < PAGE:
            break
    evs = evs[:top_n]
    if not evs:
        print("  discovery returned nothing; leaving universe untouched")
        return 0, 0

    nxt = (c.execute("SELECT COALESCE(MAX(tok),-1) FROM tokens").fetchone()[0] or -1) + 1
    n_new_ev = n_new_tok = 0
    for rank, e in enumerate(evs):
        eid, cat = e["id"], category(e)
        mk = live_markets(e)
        total = sum(len(ids) for _, _, ids, _ in mk)
        cap = sports_cap if cat == "Sports" else hard_cap
        keep, budget, capped = [], cap, 0
        for liq, m, ids, outs in mk:
            if budget - len(ids) < 0:
                capped = 1
                continue
            budget -= len(ids); keep.append((m, ids, outs))
        row = c.execute("SELECT first_seen,best_rank,max_vol24 FROM events WHERE event_id=?",
                        (eid,)).fetchone()
        vol = float(e.get("volume24hr") or 0)
        if row is None:
            n_new_ev += 1
            c.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (eid, e.get("slug"), e.get("title"), cat, e.get("endDate"),
                       now, now, rank + 1, vol, capped))
        else:
            c.execute("""UPDATE events SET last_seen=?, best_rank=MIN(best_rank,?),
                         max_vol24=MAX(max_vol24,?), capped=? WHERE event_id=?""",
                      (now, rank + 1, vol, capped, eid))
        for m, ids, outs in keep:
            cid = m["conditionId"]
            c.execute("""INSERT INTO markets VALUES(?,?,?,?,?,?,?,?,?)
                         ON CONFLICT(condition_id) DO UPDATE SET
                           last_seen=excluded.last_seen, active=excluded.active,
                           closed=excluded.closed""",
                      (cid, eid, m.get("question"), m.get("groupItemTitle"),
                       m.get("endDate"), 1, 0, now, now))
            for i, tid in enumerate(ids):
                oc = outs[i] if i < len(outs) else str(i)
                if c.execute("SELECT 1 FROM tokens WHERE token_id=?", (tid,)).fetchone():
                    c.execute("UPDATE tokens SET last_seen=?, tracking=1 WHERE token_id=?",
                              (now, tid))
                else:
                    c.execute("INSERT INTO tokens VALUES(?,?,?,?,?,?,?,1)",
                              (tid, nxt, cid, eid, oc, now, now))
                    nxt += 1; n_new_tok += 1
        if capped:
            print(f"  cap {cat:10} {total:4}->{cap:4} tokens  {(e.get('title') or '')[:46]}")
    c.commit()
    return n_new_ev, n_new_tok


def retire(c, batch=100):
    """Stop tracking tokens whose market stopped quoting. Sticky until then.

    Verified semantics of gamma /markets?condition_ids=...:
      * repeated params only -- a comma-separated list returns 0 rows silently.
      * batch caps at 100; 150 returns a loud HTTP 422 (URL length).
      * the route returns only active markets. A tracked id that comes back
        absent is active=false (closed=true/archived=true do not recover it),
        and the CLOB 404s those tokens anyway, so absent == stop tracking.
    """
    live = [r[0] for r in c.execute(
        "SELECT DISTINCT condition_id FROM tokens WHERE tracking=1").fetchall()]
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    dead = 0
    for i in range(0, len(live), batch):
        chunk = live[i:i + batch]
        q = "&".join(f"condition_ids={urllib.parse.quote(x)}" for x in chunk)
        b = get_raw(f"/markets?limit=500&{q}")
        if b is None:                      # network trouble: never retire on doubt
            print(f"  retire: batch {i // batch} failed, skipping (fail-safe)")
            continue
        got = {m.get("conditionId"): m for m in b}
        for cid in chunk:
            m = got.get(cid)
            gone = m is None or m.get("closed") or not m.get("active")
            if gone:
                c.execute("UPDATE tokens SET tracking=0,last_seen=? WHERE condition_id=?",
                          (now, cid))
                c.execute("""UPDATE markets SET closed=?,active=?,last_seen=?
                             WHERE condition_id=?""",
                          (1 if (m or {}).get("closed") else 0,
                           1 if (m or {}).get("active") else 0, now, cid))
                dead += 1
    c.commit()
    return dead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=150)
    p.add_argument("--sports-cap", type=int, default=40,
                   help="max tokens per Sports event (prop markets are the bulk)")
    p.add_argument("--hard-cap", type=int, default=128,
                   help="safety cap for any single non-sports event")
    p.add_argument("--no-retire", action="store_true")
    a = p.parse_args()
    c = init()
    print(f"[{dt.datetime.now(dt.UTC):%F %T}] refresh top-{a.top} "
          f"(sports cap {a.sports_cap}, hard cap {a.hard_cap})")
    ne, nt = refresh(c, a.top, a.sports_cap, a.hard_cap)
    dead = 0 if a.no_retire else retire(c)
    tr = c.execute("SELECT COUNT(*) FROM tokens WHERE tracking=1").fetchone()[0]
    tot = c.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
    ev = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    print(f"  +{ne} events, +{nt} tokens, retired {dead} markets")
    print(f"  universe: {tr:,} tracking / {tot:,} lifetime tokens over {ev:,} events")
    print(f"  shards at 450: {(tr + 449) // 450}  (ws cap is between 700 and 800)")


if __name__ == "__main__":
    main()
