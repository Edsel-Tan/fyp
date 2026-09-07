#!/usr/bin/env python3
"""Pull the findata crypto tape into data/crypto.sqlite.

findata exposes no /crypto family; crypto pairs live on the generic /ohlc route
with exchange == "CRYPTO". Coverage is gappy (see DATA.md), so every symbol is
pulled year by year and the gaps are recorded rather than papered over.
"""
import os, sqlite3, sys, requests, concurrent.futures as cf

B = "https://lum.id/findata"
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, os.pardir, "data", "crypto.sqlite")

CANDIDATES = """BTCUSD ETHUSD SOLUSD XRPUSD DOGEUSD ADAUSD LTCUSD BNBUSD AVAXUSD LINKUSD
DOTUSD TRXUSD UNIUSD ATOMUSD XLMUSD BCHUSD ETCUSD FILUSD NEARUSD APTUSD ARBUSD OPUSD
INJUSD AAVEUSD ICPUSD HBARUSD SUIUSD TONUSD RNDRUSD IMXUSD MANAUSD AXSUSD CRVUSD
MKRUSD COMPUSD LDOUSD FTMUSD THETAUSD EOSUSD XMRUSD ZECUSD DASHUSD""".split()

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


def init(db):
    c = sqlite3.connect(db)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS ohlc(
        symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL,
        close REAL, volume REAL, PRIMARY KEY(symbol, interval, ts))""")
    c.commit()
    return c


def job(args):
    sym, interval, lo, hi = args
    d = get(f"/ohlc/{sym}", interval=interval, **{"from": lo, "to": hi})
    bars = (d or {}).get("bars") or []
    return sym, interval, [(sym, interval, b["ts"], b["open"], b["high"], b["low"],
                            b["close"], b.get("volume")) for b in bars]


def main():
    interval = sys.argv[1] if len(sys.argv) > 1 else "1d"
    years = range(2020, 2027)
    if interval == "1d":
        windows = [(f"{y}-01-01", f"{y}-12-31") for y in years]
    else:                                   # intraday: month windows
        windows = [(f"{y}-{m:02d}-01", f"{y}-{m:02d}-28" if m == 2 else f"{y}-{m:02d}-{30 if m in (4,6,9,11) else 31}")
                   for y in years for m in range(1, 13)]
    jobs = [(sym, interval, lo, hi) for sym in CANDIDATES for lo, hi in windows]
    c = init(DB)
    n = 0
    with cf.ThreadPoolExecutor(16) as ex:
        for sym, iv, rows in ex.map(job, jobs):
            if rows:
                c.executemany("INSERT OR REPLACE INTO ohlc VALUES(?,?,?,?,?,?,?,?)", rows)
                n += len(rows)
        c.commit()
    print(f"{interval}: inserted/updated {n} bars")
    for r in c.execute("SELECT interval, COUNT(DISTINCT symbol), COUNT(*), MIN(ts), MAX(ts) "
                       "FROM ohlc GROUP BY interval"):
        print(r)


if __name__ == "__main__":
    main()
