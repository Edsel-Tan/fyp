#!/usr/bin/env python3
"""Bulk-pull 2y daily OHLC + fundamentals for the whole findata universe into SQLite.

Resumable: re-running skips symbols already marked done for a given kind.
Usage: LUMID_TOKEN=... python3 scripts/findata_pull.py [--workers 8] [--limit N]
"""
import argparse, json, os, queue, sqlite3, sys, threading, time
from datetime import date, timedelta

import requests

BASE = "https://lum.id/findata"
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "findata.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlc_daily (
  symbol TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL,
  adj_close REAL, volume REAL, vwap REAL,
  PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS fundamentals_latest (
  symbol TEXT PRIMARY KEY, period_end_date TEXT, period_type TEXT, report_date TEXT,
  currency TEXT, payload TEXT
);
CREATE TABLE IF NOT EXISTS fundamentals_history (
  symbol TEXT, period_end_date TEXT, period_type TEXT, payload TEXT,
  PRIMARY KEY (symbol, period_end_date, period_type)
);
CREATE TABLE IF NOT EXISTS symbols (symbol TEXT PRIMARY KEY, payload TEXT);
CREATE TABLE IF NOT EXISTS fetch_log (
  symbol TEXT, kind TEXT, status INTEGER, rows INTEGER, err TEXT, ts REAL,
  PRIMARY KEY (symbol, kind)
);
"""


def make_session(token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    s.headers["Accept-Encoding"] = "gzip"
    ad = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    s.mount("https://", ad)
    return s


def get(sess, path, params=None, tries=4):
    """GET with retry on 429/5xx. Returns (status, json_or_None, err)."""
    delay = 1.0
    for attempt in range(tries):
        try:
            r = sess.get(f"{BASE}{path}", params=params, timeout=60)
        except requests.RequestException as e:
            if attempt == tries - 1:
                return 0, None, f"{type(e).__name__}: {e}"
            time.sleep(delay); delay *= 2
            continue
        if r.status_code == 200:
            try:
                return 200, r.json(), None
            except ValueError as e:
                return 200, None, f"bad json: {e}"
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", delay))
            time.sleep(wait); delay *= 2
            continue
        if 500 <= r.status_code < 600:
            if attempt == tries - 1:
                return r.status_code, None, r.text[:200]
            time.sleep(delay); delay *= 2
            continue
        return r.status_code, None, r.text[:200]
    return 0, None, "retries exhausted"


def worker(sess, jobs, out, frm, to):
    while True:
        try:
            sym, kind = jobs.get_nowait()
        except queue.Empty:
            return
        if kind == "ohlc":
            st, body, err = get(sess, f"/ohlc/{sym}", {"interval": "1d", "from": frm, "to": to})
            rows = []
            if st == 200 and isinstance(body, dict):
                for b in body.get("bars") or []:
                    rows.append((sym, b.get("ts"), b.get("open"), b.get("high"), b.get("low"),
                                 b.get("close"), b.get("adj_close"), b.get("volume"), b.get("vwap")))
        elif kind == "fund_latest":
            st, body, err = get(sess, f"/fundamentals/{sym}/latest")
            rows = []
            if st == 200 and isinstance(body, dict) and body.get("period_end_date"):
                rows.append((sym, body.get("period_end_date"), body.get("period_type"),
                             body.get("report_date"), body.get("currency"), json.dumps(body)))
        else:  # fund_hist
            st, body, err = get(sess, f"/fundamentals/{sym}/history", {"period": "quarter", "limit": "12"})
            rows = []
            if st == 200 and isinstance(body, list):
                for rec in body:
                    if isinstance(rec, dict) and rec.get("period_end_date"):
                        rows.append((sym, rec.get("period_end_date"), rec.get("period_type"), json.dumps(rec)))
        out.put((sym, kind, st, rows, err))
        jobs.task_done()


def writer(conn, out, total, stop):
    done = 0
    t0 = time.time()
    pending = 0
    while not (stop.is_set() and out.empty()):
        try:
            sym, kind, st, rows, err = out.get(timeout=0.5)
        except queue.Empty:
            continue
        cur = conn.cursor()
        if kind == "ohlc" and rows:
            cur.executemany("INSERT OR REPLACE INTO ohlc_daily VALUES (?,?,?,?,?,?,?,?,?)", rows)
        elif kind == "fund_latest" and rows:
            cur.executemany("INSERT OR REPLACE INTO fundamentals_latest VALUES (?,?,?,?,?,?)", rows)
        elif kind == "fund_hist" and rows:
            cur.executemany("INSERT OR REPLACE INTO fundamentals_history VALUES (?,?,?,?)", rows)
        cur.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?,?,?)",
                    (sym, kind, st, len(rows), err, time.time()))
        done += 1
        pending += 1
        if pending >= 200:
            conn.commit(); pending = 0
        if done % 500 == 0:
            el = time.time() - t0
            rate = done / el if el else 0
            eta = (total - done) / rate if rate else 0
            print(f"  {done}/{total}  {rate:.0f}/s  eta {eta/60:.1f}m", flush=True)
    conn.commit()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N symbols (smoke test)")
    ap.add_argument("--years", type=float, default=2.0)
    args = ap.parse_args()

    token = os.environ.get("LUMID_TOKEN")
    if not token:
        sys.exit("LUMID_TOKEN not set")

    to = date.today()
    frm = to - timedelta(days=int(365.25 * args.years))
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    conn = sqlite3.connect(DB, check_same_thread=False)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit()

    sess = make_session(token)
    st, uni, err = get(sess, "/universe")
    if st != 200:
        sys.exit(f"universe fetch failed: {st} {err}")
    syms = [r["symbol"] if isinstance(r, dict) else r for r in uni]
    if args.limit:
        syms = syms[: args.limit]
    print(f"universe: {len(syms)} symbols  window {frm} -> {to}", flush=True)

    have = {(s, k) for s, k in conn.execute(
        "SELECT symbol, kind FROM fetch_log WHERE status IN (200, 404)")}
    jobs = queue.Queue()
    n = 0
    for s in syms:
        for k in ("ohlc", "fund_latest", "fund_hist"):
            if (s, k) not in have:
                jobs.put((s, k)); n += 1
    print(f"queued {n} requests ({len(have)} already done)", flush=True)
    if not n:
        print("nothing to do"); return

    out = queue.Queue(maxsize=2000)
    stop = threading.Event()
    threads = [threading.Thread(target=worker, args=(sess, jobs, out, str(frm), str(to)), daemon=True)
               for _ in range(args.workers)]
    for t in threads:
        t.start()
    wt = threading.Thread(target=lambda: writer(conn, out, n, stop), daemon=True)
    wt.start()
    for t in threads:
        t.join()
    stop.set()
    wt.join()
    conn.commit()

    print("\n--- summary ---")
    for kind, cnt, ok in conn.execute(
            "SELECT kind, COUNT(*), SUM(status=200) FROM fetch_log GROUP BY kind"):
        print(f"  {kind:12s} {cnt:6d} attempted, {ok or 0:6d} ok")
    for tbl in ("ohlc_daily", "fundamentals_latest", "fundamentals_history"):
        c = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:22s} {c:9d} rows")
    conn.close()


if __name__ == "__main__":
    main()
