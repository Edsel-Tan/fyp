#!/usr/bin/env python3
"""Re-pull the income statement at full depth.

Pass 1 fetched /fundamentals/{sym}/history with limit=12, so income statements
only reached ~3 years back while the balance sheet and cash flow (limit=80)
reach ~15. Income was therefore the binding constraint on how far a
point-in-time backtest could run: every formation date before 2024 had an empty
universe. This refills the same table with limit=80.
"""
import json, os, queue, sqlite3, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from findata_pull2 import make_session, get, DB


def main():
    token = os.environ.get("LUMID_TOKEN") or sys.exit("set LUMID_TOKEN")
    conn = sqlite3.connect(DB, check_same_thread=False, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    conn.execute("""CREATE TABLE IF NOT EXISTS fetch_log3 (
        symbol TEXT PRIMARY KEY, status INTEGER, rows INTEGER, ts REAL)""")

    syms = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM ohlc ORDER BY symbol")]
    have = {r[0] for r in conn.execute(
        "SELECT symbol FROM fetch_log3 WHERE status IN (200, 400, 404)")}
    jobs = queue.Queue()
    for s in syms:
        if s not in have:
            jobs.put(s)
    total = jobs.qsize()
    print(f"{total} symbols to refill (of {len(syms)})", flush=True)
    if not total:
        return

    out = queue.Queue(maxsize=2000)
    sess = make_session(token)

    def worker():
        while True:
            try:
                sym = jobs.get_nowait()
            except queue.Empty:
                return
            st, body, err = get(sess, f"/fundamentals/{sym}/history",
                                {"statement": "income", "period": "quarter", "limit": "80"})
            rows = []
            if st == 200 and isinstance(body, list):
                rows = [(sym, r.get("period_end_date"), r.get("period_type"), json.dumps(r))
                        for r in body if isinstance(r, dict) and r.get("period_end_date")]
            out.put((sym, st, rows))

    stop = threading.Event()

    def writer():
        done, t0 = 0, time.time()
        while not (stop.is_set() and out.empty()):
            try:
                sym, st, rows = out.get(timeout=0.5)
            except queue.Empty:
                continue
            cur = conn.cursor()
            if rows:
                cur.executemany(
                    "INSERT OR REPLACE INTO fundamentals_history VALUES (?,?,?,?)", rows)
            cur.execute("INSERT OR REPLACE INTO fetch_log3 VALUES (?,?,?,?)",
                        (sym, st, len(rows), time.time()))
            done += 1
            if done % 500 == 0:
                conn.commit()
                el = time.time() - t0
                print(f"  {done}/{total}  {done/el:.0f}/s  eta {(total-done)/(done/el)/60:.1f}m",
                      flush=True)
        conn.commit()

    wr = threading.Thread(target=writer, daemon=True); wr.start()
    ws = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
    for w in ws: w.start()
    for w in ws: w.join()
    stop.set(); wr.join()

    n, q = conn.execute(
        "SELECT COUNT(DISTINCT symbol), COUNT(*) FROM fundamentals_history").fetchone()
    print(f"\nfundamentals_history now {q:,} rows across {n:,} symbols")
    med = conn.execute("""SELECT AVG(n) FROM (SELECT COUNT(*) n FROM fundamentals_history
                          GROUP BY symbol)""").fetchone()[0]
    print(f"mean quarters per symbol: {med:.1f}")
    conn.close()


if __name__ == "__main__":
    main()
