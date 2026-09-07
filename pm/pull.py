#!/usr/bin/env python3
"""Pull the Polymarket tape findata exposes into data/pm.sqlite.

Discovery is by keyword search (there is no list-all route). Trades are paged by
time because `limit` caps at 1000 and offsets are not supported. Everything is
resumable: markets and trades are upserted by primary key.
"""
import datetime as dt, os, sqlite3, sys, requests, concurrent.futures as cf

B = "https://lum.id/findata"
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, os.pardir, "data", "pm.sqlite")
TAPE_START = dt.datetime(2026, 5, 21, tzinfo=dt.timezone.utc)

QUERIES = """trump fed election bitcoin ethereum nba nfl mlb israel ukraine china russia
ai openai google inflation cpi jobs rate recession gdp oil gold tariff supreme court
senate house governor mayor poll debate war ceasefire nuclear iran venezuela nato
oscar grammy emmy movie box super bowl world cup soccer premier tennis golf ufc boxing
launch spacex tesla apple nvidia microsoft amazon meta stock ipo merger crypto solana
xrp doge etf sec regulation shutdown impeach resign nominee cabinet weather hurricane
temperature earthquake covid vaccine drug fda approval""".split()

s = requests.Session()
s.headers["Authorization"] = f"Bearer {os.environ['LUMID_TOKEN']}"
s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=24, pool_maxsize=24))


def get(path, **kw):
    for _ in range(4):
        try:
            r = s.get(B + path, params=kw, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (400, 404):
                return None
        except Exception:
            pass
    return None


def init():
    c = sqlite3.connect(DB)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS markets(
        market_id TEXT PRIMARY KEY, title TEXT, slug TEXT, volume REAL,
        start_date TEXT, end_date TEXT, closed INT, detail TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS trades(
        trade_id TEXT PRIMARY KEY, market_id TEXT, token_id TEXT, side TEXT,
        price REAL, size REAL, taker TEXT, ts TEXT)""")
    c.execute("CREATE INDEX IF NOT EXISTS ix_tr_mkt ON trades(market_id)")
    c.execute("""CREATE TABLE IF NOT EXISTS pulled(market_id TEXT PRIMARY KEY, n INT)""")
    c.commit()
    return c


def discover(c):
    found = {}
    def q(term):
        d = get("/prediction-markets/markets/search", q=term, limit=200)
        return [m for m in (d or []) if m.get("venue") == "polymarket"]
    with cf.ThreadPoolExecutor(12) as ex:
        for ms in ex.map(q, QUERIES):
            for m in ms:
                found[m["market_id"]] = m
    rows = [(m["market_id"], m.get("title"), m.get("slug"), m.get("volume"),
             m.get("start_date"), m.get("end_date"), int(bool(m.get("closed"))), None)
            for m in found.values()]
    c.executemany("INSERT OR IGNORE INTO markets VALUES(?,?,?,?,?,?,?,?)", rows)
    c.commit()
    print(f"discovered {len(found)} polymarket markets "
          f"({c.execute('SELECT COUNT(*) FROM markets').fetchone()[0]} total in db)")


def tape(market_id, lo, hi, step_days=3):
    """Page the trade tape by time; the route caps at 1000 rows per call."""
    out, cur = {}, lo
    while cur < hi:
        nxt = min(cur + dt.timedelta(days=step_days), hi)
        d = get(f"/prediction-markets/trades/polymarket/{market_id}",
                limit=1000,
                **{"from": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "to": nxt.strftime("%Y-%m-%dT%H:%M:%SZ")})
        if isinstance(d, list):
            for t in d:
                out[t["trade_id"]] = t
            if len(d) >= 1000 and step_days > 0.05:           # saturated: bisect
                out.update(tape(market_id, cur, nxt, step_days / 3))
        cur = nxt
    return out


def pull_trades(c, limit=None, workers=10, min_vol=100_000):
    """Only markets that both resolve inside the tape window and carry real volume.

    A market whose end_date falls outside the window can never be scored against
    its outcome, and one below `min_vol` contributes almost no traders."""
    todo = c.execute(
        "SELECT m.market_id, m.start_date, m.end_date FROM markets m "
        "LEFT JOIN pulled p USING(market_id) WHERE p.market_id IS NULL "
        "AND m.volume > ? AND substr(m.end_date,1,10) BETWEEN '2026-05-25' AND '2026-09-05' "
        "ORDER BY m.volume DESC", (min_vol,)).fetchall()
    if limit:
        todo = todo[:limit]
    print(f"pulling trades for {len(todo)} markets", flush=True)
    now = dt.datetime.now(dt.timezone.utc)

    def parse(x, default):
        try:
            return dt.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        except Exception:
            return default

    def job(row):
        mid, sd, ed = row
        lo = max(TAPE_START, parse(sd, TAPE_START))
        hi = min(now, parse(ed, now) + dt.timedelta(days=1))
        return mid, (tape(mid, lo, hi) if hi > lo else {})

    n_done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for mid, tr in ex.map(job, todo):
            rows = [(t["trade_id"], mid, t.get("token_id"), t.get("side"),
                     t.get("price"), t.get("size"), t.get("taker"), t.get("ts"))
                    for t in tr.values()]
            if rows:
                c.executemany("INSERT OR IGNORE INTO trades VALUES(?,?,?,?,?,?,?,?)", rows)
            c.execute("INSERT OR REPLACE INTO pulled VALUES(?,?)", (mid, len(rows)))
            n_done += 1
            if n_done % 20 == 0:
                c.commit()
                print(f"  {n_done}/{len(todo)}  last={len(rows)} trades", flush=True)
    c.commit()
    print("trades in db:", c.execute("SELECT COUNT(*) FROM trades").fetchone()[0])


def pull_detail(c, workers=12):
    todo = [r[0] for r in c.execute(
        "SELECT market_id FROM markets WHERE detail IS NULL AND market_id IN "
        "(SELECT market_id FROM pulled WHERE n > 0)")]
    print(f"fetching detail for {len(todo)} markets")
    import json
    def job(mid):
        return mid, get(f"/prediction-markets/markets/polymarket/{mid}")
    ok = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for mid, d in ex.map(job, todo):
            if d:
                ok += 1
                c.execute("UPDATE markets SET detail=? WHERE market_id=?", (json.dumps(d), mid))
    c.commit()
    print(f"detail resolved for {ok}/{len(todo)}")


if __name__ == "__main__":
    c = init()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "discover"):
        discover(c)
    if what in ("all", "trades"):
        pull_trades(c, limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    if what in ("all", "detail"):
        pull_detail(c)
