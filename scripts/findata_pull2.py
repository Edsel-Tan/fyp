#!/usr/bin/env python3
"""Second-pass pull: everything the first pull missed.

The first pull (findata_pull.py) took 2 years of daily bars and the income
statement only. That left four holes that materially limited the analysis:

  deep history  -- /ohlc actually serves back to ~2010, not 2 years. Two years
                   is one regime; you cannot estimate a long-run CAGR from it.
  dividends     -- adj_close is NULL for every row, so returns were price-only.
                   /dividends/{sym} carries the full cash record.
  balance sheet -- fundamentals_history was income-statement only, so no
                   point-in-time ROE, leverage or FCF at a past formation date.
  benchmarks    -- the universe roster excludes ETFs and indices entirely.

Resumable: re-running skips (symbol, kind) pairs already logged 200 or 404.
Usage: LUMID_TOKEN=... python3 scripts/findata_pull2.py [--workers 8]
"""
import argparse, json, os, queue, sqlite3, sys, threading, time
from datetime import date

import requests

BASE = "https://lum.id/findata"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "findata.sqlite")

START = "2006-01-01"
BENCHMARKS = ["SPY", "QQQ", "IWM", "VTI", "VOO", "DIA", "RSP", "^GSPC", "^NDX", "^RUT", "AGG", "TLT"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlc (
  symbol TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
  PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS dividends (
  symbol TEXT, ex_date TEXT, payment_date TEXT, amount REAL, adj_amount REAL,
  frequency TEXT, PRIMARY KEY (symbol, ex_date, amount)
);
CREATE TABLE IF NOT EXISTS splits (
  symbol TEXT, date TEXT, ratio REAL, PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS fund_balance (
  symbol TEXT, period_end_date TEXT, period_type TEXT, payload TEXT,
  PRIMARY KEY (symbol, period_end_date)
);
CREATE TABLE IF NOT EXISTS fund_cashflow (
  symbol TEXT, period_end_date TEXT, period_type TEXT, payload TEXT,
  PRIMARY KEY (symbol, period_end_date)
);
CREATE TABLE IF NOT EXISTS fetch_log2 (
  symbol TEXT, kind TEXT, status INTEGER, rows INTEGER, err TEXT, ts REAL,
  PRIMARY KEY (symbol, kind)
);
"""

KINDS = ["ohlc", "div", "split", "bal", "cf"]


def make_session(token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    s.headers["Accept-Encoding"] = "gzip"
    ad = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    s.mount("https://", ad)
    return s


def get(sess, path, params=None, tries=4):
    delay = 1.0
    for attempt in range(tries):
        try:
            r = sess.get(f"{BASE}{path}", params=params, timeout=90)
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
            time.sleep(float(r.headers.get("Retry-After", delay))); delay *= 2
            continue
        if 500 <= r.status_code < 600:
            if attempt == tries - 1:
                return r.status_code, None, r.text[:200]
            time.sleep(delay); delay *= 2
            continue
        return r.status_code, None, r.text[:200]
    return 0, None, "retries exhausted"


def worker(sess, jobs, out, to):
    while True:
        try:
            sym, kind = jobs.get_nowait()
        except queue.Empty:
            return
        rows = []
        if kind == "ohlc":
            st, body, err = get(sess, f"/ohlc/{sym}", {"interval": "1d", "from": START, "to": to})
            if st == 200 and isinstance(body, dict):
                rows = [(sym, b.get("ts"), b.get("open"), b.get("high"), b.get("low"),
                         b.get("close"), b.get("volume")) for b in (body.get("bars") or [])]
        elif kind == "div":
            st, body, err = get(sess, f"/dividends/{sym}", {"limit": "5000"})
            if st == 200 and isinstance(body, list):
                seen = set()
                for d in body:
                    if not isinstance(d, dict) or not d.get("date"):
                        continue
                    # the feed repeats an ex-date with null yield/frequency; the
                    # primary key collapses those, keep the richer record first
                    k = (d.get("date"), d.get("amount"))
                    if k in seen:
                        continue
                    seen.add(k)
                    rows.append((sym, d.get("date"), d.get("payment_date"), d.get("amount"),
                                 d.get("adj_amount"), d.get("frequency")))
        elif kind == "split":
            st, body, err = get(sess, f"/splits/{sym}")
            if st == 200 and isinstance(body, list):
                rows = [(sym, s.get("date"), s.get("ratio")) for s in body
                        if isinstance(s, dict) and s.get("date")]
        elif kind in ("bal", "cf"):
            stmt = "balance" if kind == "bal" else "cashflow"
            st, body, err = get(sess, f"/fundamentals/{sym}/history",
                                {"statement": stmt, "period": "quarter", "limit": "80"})
            if st == 200 and isinstance(body, list):
                rows = [(sym, r.get("period_end_date"), r.get("period_type"), json.dumps(r))
                        for r in body if isinstance(r, dict) and r.get("period_end_date")]
        out.put((sym, kind, st, rows, err))
        jobs.task_done()


TABLE = {"ohlc": ("ohlc", 7), "div": ("dividends", 6), "split": ("splits", 3),
         "bal": ("fund_balance", 4), "cf": ("fund_cashflow", 4)}


def writer(conn, out, total, stop):
    done, t0 = 0, time.time()
    while not (stop.is_set() and out.empty()):
        try:
            sym, kind, st, rows, err = out.get(timeout=0.5)
        except queue.Empty:
            continue
        cur = conn.cursor()
        if rows:
            tbl, n = TABLE[kind]
            cur.executemany(f"INSERT OR REPLACE INTO {tbl} VALUES ({','.join('?'*n)})", rows)
        cur.execute("INSERT OR REPLACE INTO fetch_log2 VALUES (?,?,?,?,?,?)",
                    (sym, kind, st, len(rows), err, time.time()))
        done += 1
        if done % 400 == 0:
            conn.commit()
            el = time.time() - t0
            print(f"  {done}/{total}  {done/el:.0f} req/s  eta {(total-done)/(done/el)/60:.1f}m",
                  flush=True)
        out.task_done()
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    token = os.environ.get("LUMID_TOKEN")
    if not token:
        sys.exit("set LUMID_TOKEN")

    conn = sqlite3.connect(DB, check_same_thread=False, timeout=120)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=120000")

    # only symbols that actually returned bars in pass 1, plus benchmarks
    syms = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM ohlc_daily ORDER BY symbol")]
    syms = BENCHMARKS + [s for s in syms if s not in set(BENCHMARKS)]
    if a.limit:
        syms = syms[:a.limit]

    have = {(s, k) for s, k in conn.execute(
        "SELECT symbol, kind FROM fetch_log2 WHERE status IN (200, 404, 400)")}
    jobs = queue.Queue()
    for s in syms:
        for k in KINDS:
            if (s, k) not in have:
                jobs.put((s, k))
    total = jobs.qsize()
    print(f"{len(syms)} symbols, {total} requests queued "
          f"({len(have)} already done), window {START} -> today", flush=True)
    if not total:
        return

    out = queue.Queue(maxsize=2000)
    stop = threading.Event()
    sess = make_session(token)
    to = date.today().isoformat()
    ws = [threading.Thread(target=worker, args=(sess, jobs, out, to), daemon=True)
          for _ in range(a.workers)]
    wr = threading.Thread(target=writer, args=(conn, out, total, stop), daemon=True)
    wr.start()
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    stop.set()
    wr.join()
    conn.commit()

    print("\n--- results ---")
    for k in KINDS:
        for st, n, rows in conn.execute(
                "SELECT status, COUNT(*), SUM(rows) FROM fetch_log2 WHERE kind=? "
                "GROUP BY status ORDER BY COUNT(*) DESC", (k,)):
            print(f"  {k:6s} status {st:4d}  {n:6d} symbols  {rows or 0:>12,} rows")
    conn.close()


if __name__ == "__main__":
    main()
