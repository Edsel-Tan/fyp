#!/usr/bin/env python3
"""Read-only dashboard for the Polymarket live capture, served on :2217.

Answers the two questions a long capture actually raises: *is it still healthy*
(coverage, shards, drops, gaps) and *what is in there* (which events, which
markets, what did the book do).  Everything is derived from data/pmlive.sqlite
and the parquet bars -- the dashboard never writes, so it is safe to run against
a live capture.
"""
import argparse, glob, json, os, re, sqlite3, time, datetime as dt
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
DB = os.path.join(ROOT, "data", "pmlive.sqlite")
OUT = os.path.join(ROOT, "data", "pmlive")
BARS = os.path.join(OUT, "bars")
HEALTH = os.path.join(OUT, "health.log")
PAGE = os.path.join(HERE, "pmlive_dash.html")

HL = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) shards=(\d+) sub=(\d+) books=(\d+) "
                r"cov=([\d.]+)% msgs=(\d+) drops=(\d+)")


def db():
    if not os.path.exists(DB):
        return None
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=5)
    c.row_factory = sqlite3.Row
    return c


def bar_files(hours=None):
    fs = sorted(glob.glob(os.path.join(BARS, "dt=*", "bars-*.parquet")))
    if hours is None:
        return fs
    cut = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours + 1)
    keep = []
    for f in fs:
        try:
            d = os.path.basename(os.path.dirname(f)).split("=", 1)[1]
            h = int(os.path.basename(f).split("-")[1].split(".")[0])
            when = dt.datetime.strptime(d, "%Y-%m-%d").replace(hour=h, tzinfo=dt.UTC)
            if when >= cut:
                keep.append(f)
        except Exception:
            keep.append(f)
    return keep


def read_health(hours=24, limit=3000):
    if not os.path.exists(HEALTH):
        return []
    cut = time.time() - hours * 3600
    rows = []
    with open(HEALTH) as fh:
        for line in fh:
            m = HL.match(line)
            if not m:
                continue
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.UTC)
            if t.timestamp() < cut:
                continue
            rows.append({"t": int(t.timestamp()), "shards": int(m.group(2)),
                         "sub": int(m.group(3)), "books": int(m.group(4)),
                         "cov": float(m.group(5)), "msgs": int(m.group(6)),
                         "drops": int(m.group(7))})
    if len(rows) > limit:                      # thin uniformly, keep the ends
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)] + rows[-1:]
    return rows


def api_status():
    c = db()
    out = {"tracking": 0, "lifetime": 0, "events": 0, "markets": 0, "capped": 0,
           "rows": 0, "bytes": 0, "files": 0, "first": None, "last": None,
           "health": None, "running": False, "disk_free": None}
    if c:
        out["tracking"] = c.execute("SELECT COUNT(*) FROM tokens WHERE tracking=1").fetchone()[0]
        out["lifetime"] = c.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
        out["events"] = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        out["markets"] = c.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
        out["capped"] = c.execute("SELECT COUNT(*) FROM events WHERE capped=1").fetchone()[0]
        c.close()
    fs = bar_files()
    out["files"] = len(fs)
    out["bytes"] = sum(os.path.getsize(f) for f in fs) + \
        sum(os.path.getsize(f) for f in glob.glob(os.path.join(OUT, "trades", "dt=*", "*.parquet")))
    if fs:
        try:
            import pyarrow.parquet as pq
            out["rows"] = sum(pq.ParquetFile(f).metadata.num_rows for f in fs)
            lo = pq.ParquetFile(fs[0]).metadata.row_group(0).column(1).statistics
            hi = pq.ParquetFile(fs[-1]).metadata
            hi = hi.row_group(hi.num_row_groups - 1).column(1).statistics
            out["first"], out["last"] = int(lo.min) * 60, int(hi.max) * 60
        except Exception:
            pass
    h = read_health(hours=2)
    if h:
        out["health"] = h[-1]
        out["running"] = (time.time() - h[-1]["t"]) < 180
    try:
        st = os.statvfs(ROOT)
        out["disk_free"] = st.f_bavail * st.f_frsize
    except Exception:
        pass
    return out


def api_events(q="", limit=300, sort="rank"):
    c = db()
    if not c:
        return []
    order = {"rank": "e.best_rank ASC", "vol": "e.max_vol24 DESC",
             "tokens": "ntok DESC", "new": "e.first_seen DESC"}.get(sort, "e.best_rank ASC")
    sql = f"""SELECT e.event_id, e.title, e.category, e.best_rank, e.max_vol24,
                     e.first_seen, e.end_date, e.capped,
                     COUNT(DISTINCT t.token_id) AS ntok,
                     SUM(CASE WHEN t.tracking=1 THEN 1 ELSE 0 END) AS nlive,
                     COUNT(DISTINCT t.condition_id) AS nmkt
              FROM events e LEFT JOIN tokens t ON t.event_id = e.event_id
              {"WHERE LOWER(e.title) LIKE ?" if q else ""}
              GROUP BY e.event_id ORDER BY {order} LIMIT ?"""
    args = ([f"%{q.lower()}%"] if q else []) + [limit]
    rows = [dict(r) for r in c.execute(sql, args).fetchall()]
    c.close()
    return rows


def api_categories():
    c = db()
    if not c:
        return []
    rows = [dict(r) for r in c.execute(
        """SELECT e.category AS category, COUNT(DISTINCT e.event_id) AS events,
                  COUNT(t.token_id) AS tokens
           FROM events e LEFT JOIN tokens t ON t.event_id=e.event_id
           GROUP BY e.category ORDER BY tokens DESC""").fetchall()]
    c.close()
    return rows


def api_event(eid):
    c = db()
    if not c:
        return {}
    ev = c.execute("SELECT * FROM events WHERE event_id=?", (eid,)).fetchone()
    if not ev:
        c.close()
        return {}
    mk = {}
    for r in c.execute("""SELECT m.condition_id, m.question, m.group_title, m.closed,
                                 t.tok, t.outcome, t.tracking
                          FROM markets m JOIN tokens t ON t.condition_id=m.condition_id
                          WHERE m.event_id=? ORDER BY m.condition_id, t.tok""",
                       (eid,)).fetchall():
        d = mk.setdefault(r["condition_id"], {"condition_id": r["condition_id"],
                                              "question": r["question"],
                                              "group_title": r["group_title"],
                                              "closed": r["closed"], "tokens": []})
        d["tokens"].append({"tok": r["tok"], "outcome": r["outcome"],
                            "tracking": r["tracking"]})
    c.close()
    return {"event": dict(ev), "markets": list(mk.values())}


def api_series(toks, hours=24, maxpts=1200):
    fs = bar_files(hours)
    if not fs or not toks:
        return {"series": []}
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
    cut = int((time.time() - hours * 3600) // 60)
    parts = []
    for f in fs:
        try:
            t = pq.read_table(f, columns=["tok", "ts", "c", "bb", "ba", "n"])
        except Exception:
            continue
        m = pc.and_(pc.is_in(t["tok"], value_set=pa.array(toks, pa.int32())),
                    pc.greater_equal(t["ts"], cut))
        t = t.filter(m)
        if t.num_rows:
            parts.append(t)
    if not parts:
        return {"series": []}
    t = pa.concat_tables(parts).sort_by([("tok", "ascending"), ("ts", "ascending")])
    d = t.to_pydict()
    by = {}
    for tok, ts, c_, bb, ba, n in zip(d["tok"], d["ts"], d["c"], d["bb"], d["ba"], d["n"]):
        by.setdefault(tok, []).append([ts * 60, c_ / 1000.0, bb / 1000.0, ba / 1000.0, n])
    out = []
    for tok, pts in by.items():
        if len(pts) > maxpts:
            step = len(pts) / maxpts
            pts = [pts[int(i * step)] for i in range(maxpts)] + pts[-1:]
        out.append({"tok": tok, "points": pts})
    return {"series": out}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj):
        self._send(200, json.dumps(obj, default=str), "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        one = lambda k, d=None: (q.get(k) or [d])[0]
        try:
            if u.path in ("/", "/index.html"):
                with open(PAGE, "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")
            if u.path == "/api/status":
                return self._json(api_status())
            if u.path == "/api/health":
                return self._json(read_health(int(one("hours", 24))))
            if u.path == "/api/categories":
                return self._json(api_categories())
            if u.path == "/api/events":
                return self._json(api_events(one("q", "") or "",
                                             min(int(one("limit", 300)), 2000),
                                             one("sort", "rank")))
            if u.path == "/api/event":
                return self._json(api_event(one("id", "")))
            if u.path == "/api/series":
                toks = [int(x) for x in (one("toks", "") or "").split(",") if x.strip()]
                return self._json(api_series(toks[:3], int(one("hours", 24))))
            self._send(404, "not found", "text/plain")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}), "application/json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=2217)
    p.add_argument("--host", default="0.0.0.0",
                   help="0.0.0.0 exposes on the LAN; use 127.0.0.1 to keep it local")
    a = p.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), H)
    srv.daemon_threads = True
    print(f"pmlive dashboard on http://{a.host}:{a.port}  (reading {OUT})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
