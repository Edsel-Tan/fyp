#!/usr/bin/env python3
"""Load symbol metadata (/symbols batches) and the actively-trading list into findata.sqlite."""
import json, os, sqlite3, sys, time

import requests

BASE = "https://lum.id/findata"
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "findata.sqlite")
BATCH = 500

token = os.environ.get("LUMID_TOKEN") or sys.exit("LUMID_TOKEN not set")
s = requests.Session()
s.headers["Authorization"] = f"Bearer {token}"

conn = sqlite3.connect(DB, timeout=60)
conn.execute("PRAGMA busy_timeout=60000")
conn.executescript("""
CREATE TABLE IF NOT EXISTS symbols (symbol TEXT PRIMARY KEY, payload TEXT);
CREATE TABLE IF NOT EXISTS actively_trading (symbol TEXT PRIMARY KEY, name TEXT, snapshot_date TEXT);
""")

uni = s.get(f"{BASE}/universe", timeout=60).json()
syms = [r["symbol"] if isinstance(r, dict) else r for r in uni]
print(f"universe: {len(syms)}")

got = 0
for i in range(0, len(syms), BATCH):
    chunk = syms[i:i + BATCH]
    for attempt in range(4):
        r = s.get(f"{BASE}/symbols", params={"ticker": ",".join(chunk)}, timeout=90)
        if r.status_code == 200:
            break
        time.sleep(2 ** attempt)
    else:
        print(f"  batch {i} failed: {r.status_code}")
        continue
    body = r.json()
    rows = body.get("data", body) if isinstance(body, dict) else body
    recs = [(x["symbol"], json.dumps(x)) for x in rows if isinstance(x, dict) and x.get("symbol")]
    conn.executemany("INSERT OR REPLACE INTO symbols VALUES (?,?)", recs)
    conn.commit()
    got += len(recs)
    print(f"  {i + len(chunk)}/{len(syms)} -> {got} metadata rows", flush=True)

r = s.get(f"{BASE}/universe/actively-trading", params={"limit": 100000}, timeout=180)
body = r.json()
rows = body.get("data", body) if isinstance(body, dict) else body
conn.executemany("INSERT OR REPLACE INTO actively_trading VALUES (?,?,?)",
                 [(x["symbol"], x.get("name"), x.get("snapshot_date")) for x in rows])
conn.commit()
print(f"actively_trading: {len(rows)}")
conn.close()
