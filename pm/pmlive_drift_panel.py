#!/usr/bin/env python3
"""Rebuild the text panel for the months after the Polymarket-v1 archive ends.

This is the data side of pm/PREREG_drift.md, and it exists only to serve that one
pre-registered test: the archive stops on 2026-04-28, so the 24-hour drift arm of
RESULTS.md sec.3.6 can only be tested on untouched months if those months are rebuilt
from Polymarket's public APIs. Everything here copies a construction that already
exists, and says which:

  markets    Gamma, resolved, closed on or after 2026-05-01, not in the archive, and
             `nl` under pmv1_text_panel's own RE_UNIX / RE_ISO
  fills      data-api /trades, taker-only (its default). Its offset cap is 10,500, so
             each market is paged backwards with `end=` until the history is exhausted
  p_event, D on outcome index 0, as pmv1_prep.py's p_event convention
  bars       pmv1_spread.BARS: $100k lifetime filter, per-market 90th-pct large trade,
             last p_event, signed and absolute flow, taker-side VWAPs
  rows       pmv1_text_panel.PANEL, the same market-day aggregation, y_fwd at +24 h
  ids        duckdb hash(condition_id) >> 1, so market_id / event_id match the archive's

    python pm/pmlive_drift_panel.py crawl       # Gamma + data-api -> data/prereg/fills/
    python pm/pmlive_drift_panel.py build       # fills -> data/prereg/panel.parquet
"""
import argparse, concurrent.futures as cf, json, os, re, sys, time, urllib.parse, urllib.request
import duckdb, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
NLP = os.path.join(ROOT, "data", "nlp")
OUT = os.path.join(ROOT, "data", "prereg")
FILLS = os.path.join(OUT, "fills")
sys.path.insert(0, HERE)
from pmv1_text_panel import RE_UNIX, RE_ISO

GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
START = "2026-05-01T00:00:00Z"
MIN_VOLUME = 100_000          # pmv1_prep.py --min-volume
LARGE_Q = 0.90                # pmv1_prep.py --large-q
H = 24                        # forward horizon, hours


def get(url, tries=6):
    for k in range(tries):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": "fyp-research/1.0"})
            return json.load(urllib.request.urlopen(r, timeout=60))
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


# ------------------------------------------------------------------- markets
def gamma_markets():
    """Every closed market whose scheduled end is on or after START.

    Gamma refuses offsets past a few thousand (HTTP 422), so page by time: sort on
    endDate, restart each window at the last endDate seen, dedup on conditionId.
    """
    rows, seen, lo = [], set(), START
    while True:
        off, got_new, last = 0, 0, lo
        while off < 2000:
            q = urllib.parse.urlencode(dict(closed="true", limit=100, offset=off,
                                            end_date_min=lo, order="endDate", ascending="true",
                                            # server-side pre-screen; see select()
                                            volume_num_min=MIN_VOLUME // 2))
            page = get(f"{GAMMA}/markets?{q}")
            if not page:
                break
            for m in page:
                if m.get("conditionId") not in seen:
                    seen.add(m.get("conditionId")); rows.append(m); got_new += 1
                last = max(last, m.get("endDate") or last)
            off += len(page)
            if len(page) < 100:
                break
        print(f"  gamma {len(rows)} markets, through {last}", flush=True)
        if got_new == 0 or off < 2000:
            return rows
        lo = last


def is_nl(slug):
    return not (re.search(RE_UNIX, slug) or re.search(RE_ISO, slug))


def select(ms):
    """Resolved, binary-priced, closed after START, `nl`, and plausibly >= $100k.

    `volumeNum` is Polymarket's lifetime figure; the exact $100k taker-notional filter
    is applied to the fills in build(). The pre-screen only saves API calls, so it is
    set at half the threshold to be sure it never binds before the real filter does.
    """
    keep = []
    for m in ms:
        try:
            px = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
        except ValueError:
            continue
        if (m.get("umaResolutionStatus") != "resolved" or len(px) != 2
                or sorted(px) != [0.0, 1.0] or not m.get("conditionId")
                or (m.get("closedTime") or "") < START[:10]
                or float(m.get("volumeNum") or 0) < MIN_VOLUME / 2
                or not is_nl(m.get("slug") or "")):
            continue
        ev = (m.get("events") or [{}])[0]
        keep.append(dict(condition_id=m["conditionId"].lower(), slug=m["slug"],
                         neg_risk=bool(m.get("negRisk")),
                         neg_risk_market_id=(m.get("negRiskMarketID") or "").lower() or None,
                         end_date=m.get("endDate"), closed_time=m.get("closedTime"),
                         win_event=px[0], volume=float(m.get("volumeNum") or 0),
                         event_gamma_id=ev.get("id")))
    return pd.DataFrame(keep)


def event_tags(ids):
    out = {}
    def one(i):
        try:
            e = get(f"{GAMMA}/events/{i}")
            return i, [t.get("label") for t in (e.get("tags") or []) if t.get("label")]
        except Exception:
            return i, []
    with cf.ThreadPoolExecutor(8) as ex:
        for i, tags in ex.map(one, ids):
            out[i] = tags
    return out


# --------------------------------------------------------------------- fills
def fills(cid):
    """A market's complete taker-fill history, newest first, paged back with `end=`."""
    seen, rows, end = set(), [], None
    while True:
        q = dict(market=cid, limit=500)
        page_all = []
        for off in range(0, 10_001, 500):
            q.update(offset=off)
            if end is not None:
                q["end"] = end
            page = get(f"{DATA}/trades?{urllib.parse.urlencode(q)}")
            page_all += page
            if len(page) < 500:
                break
        new = 0
        for t in page_all:
            k = (t.get("transactionHash"), t.get("asset"), t.get("proxyWallet"),
                 t.get("side"), t.get("size"), t.get("price"), t.get("timestamp"))
            if k in seen:
                continue
            seen.add(k); new += 1
            rows.append(dict(ts=int(t["timestamp"]), idx=int(t["outcomeIndex"]),
                             side=t["side"], size=float(t["size"]), price=float(t["price"])))
        if len(page_all) < 10_500 or new == 0:
            return rows
        end = min(int(t["timestamp"]) for t in page_all)   # inclusive; dedup handles it


def crawl(a):
    os.makedirs(FILLS, exist_ok=True)
    mpath = os.path.join(OUT, "markets.parquet")
    if os.path.exists(mpath):
        mk = pd.read_parquet(mpath)
    else:
        mk = select(gamma_markets())
        tags = event_tags(sorted(set(mk.event_gamma_id.dropna())))
        mk["tags"] = mk.event_gamma_id.map(lambda i: json.dumps(tags.get(i, [])))
        con = duckdb.connect()
        con.register("mk", mk)
        mk = con.sql("""SELECT *, (hash(condition_id) >> 1)::BIGINT AS market_id,
                        (hash(COALESCE(neg_risk_market_id, condition_id)) >> 1)::BIGINT AS event_id
                        FROM mk""").df()
        seen = duckdb.sql(f"SELECT market_id, event_id FROM read_parquet('{NLP}/resolution.parquet')").df()
        n0 = len(mk)
        mk = mk[~mk.market_id.isin(seen.market_id) & ~mk.event_id.isin(seen.event_id)]
        print(f"candidates {n0}, after dropping archive markets/events {len(mk)}")
        mk.to_parquet(mpath)
    todo = [c for c in mk.condition_id if not os.path.exists(os.path.join(FILLS, f"{c}.parquet"))]
    print(f"{len(mk)} markets, {len(todo)} to fetch", flush=True)
    fail = []
    def one(c):
        try:
            f = pd.DataFrame(fills(c), columns=["ts", "idx", "side", "size", "price"])
            f.to_parquet(os.path.join(FILLS, f"{c}.parquet"))
            return c, len(f), None
        except Exception as e:
            return c, 0, repr(e)
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for i, (c, n, err) in enumerate(ex.map(one, todo)):
            if err:
                fail.append((c, err))
            if i % 50 == 0 or err:
                print(f"  {i+1}/{len(todo)} {c[:10]} {n} fills {err or ''}", flush=True)
    json.dump(fail, open(os.path.join(OUT, "crawl_failures.json"), "w"), indent=1)
    print(f"done; {len(fail)} failures (dropped before any feature is computed)")


# --------------------------------------------------------------------- build
BARS = """
CREATE TABLE bars AS
WITH src AS (
  SELECT condition_id, ts AS block_timestamp,
         CASE WHEN idx = 0 THEN price ELSE 1 - price END              AS p_event,
         CASE WHEN (idx = 0) = (side = 'BUY') THEN 1 ELSE -1 END       AS D,
         size * price                                                  AS usdc_amount
  FROM fills WHERE size > 0 AND price > 0 AND price < 1
), src2 AS (SELECT * FROM src WHERE p_event > 0 AND p_event < 1 AND usdc_amount > 0),
mkt AS (
  SELECT condition_id, quantile_cont(usdc_amount, {q}) AS thr
  FROM src2 GROUP BY 1 HAVING sum(usdc_amount) >= {minv}
), tagged AS (
  SELECT s.*, (s.block_timestamp / 3600)::BIGINT AS bar, s.usdc_amount >= m.thr AS is_large
  FROM src2 s JOIN mkt m USING (condition_id)
)
SELECT (hash(condition_id) >> 1)::BIGINT AS market_id, bar, NULL::VARCHAR AS cat,
       sum(usdc_amount) AS gross,
       arg_max(p_event, block_timestamp) AS p,
       sum(CASE WHEN is_large THEN D * usdc_amount ELSE 0 END) AS sgn_large,
       sum(CASE WHEN is_large THEN usdc_amount ELSE 0 END)     AS abs_large,
       sum(CASE WHEN NOT is_large THEN D * usdc_amount ELSE 0 END) AS sgn_small,
       sum(CASE WHEN NOT is_large THEN usdc_amount ELSE 0 END)     AS abs_small,
       sum(CASE WHEN D = 1 THEN p_event * usdc_amount END)
         / nullif(sum(CASE WHEN D = 1 THEN usdc_amount END), 0)   AS vwap_buy,
       sum(CASE WHEN D = -1 THEN p_event * usdc_amount END)
         / nullif(sum(CASE WHEN D = -1 THEN usdc_amount END), 0)  AS vwap_sell,
       -- present in data/pmv1_spreadbars_1h.parquet (buy + sell = gross there to 1e-16)
       sum(CASE WHEN D = 1 THEN usdc_amount ELSE 0 END)  AS buy_notional,
       sum(CASE WHEN D = -1 THEN usdc_amount ELSE 0 END) AS sell_notional
FROM tagged GROUP BY 1, 2
"""


def build(a):
    from pmv1_text_panel import PANEL
    mk = pd.read_parquet(os.path.join(OUT, "markets.parquet"))
    con = duckdb.connect()
    con.execute(f"CREATE TABLE fills AS SELECT regexp_extract(filename, '([^/]+)\\.parquet$', 1) "
                f"AS condition_id, * EXCLUDE (filename) FROM read_parquet('{FILLS}/*.parquet', filename=true)")
    con.execute(BARS.format(q=LARGE_Q, minv=MIN_VOLUME))
    con.register("mk", mk)
    # the same three inputs PANEL reads, in the same shapes
    con.execute("CREATE TABLE slugs AS SELECT market_id, slug, tags AS cat, 0 AS fills FROM mk")
    con.execute("""CREATE TABLE res AS SELECT market_id, event_id, win_event, neg_risk,
                   epoch(end_date::TIMESTAMPTZ)::BIGINT AS close_ts FROM mk""")
    # PANEL reads three files by fixed names; write ours under those names in one
    # directory and point both of its path placeholders at it
    d = os.path.join(OUT, "tmp"); os.makedirs(d, exist_ok=True)
    con.execute(f"COPY bars TO '{d}/pmv1_spreadbars_1h.parquet' (FORMAT parquet)")
    con.execute(f"COPY slugs TO '{d}/slugs.parquet' (FORMAT parquet)")
    con.execute(f"COPY res TO '{d}/resolution.parquet' (FORMAT parquet)")
    df = con.sql(PANEL.format(nlp=d, data=d, h=H)).df()
    df["text"] = df.slug.str.replace("-", " ", regex=False)
    df["kind"] = "nl"
    df.rename(columns={"cat": "tags"}, inplace=True)
    df.to_parquet(os.path.join(OUT, "panel.parquet"))
    print(f"panel {len(df):,} rows, {df.market_id.nunique():,} markets, "
          f"{df.event_id.nunique():,} events, y_fwd non-null {df.y_fwd.notna().sum():,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["crawl", "build"])
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    crawl(a) if a.cmd == "crawl" else build(a)


if __name__ == "__main__":
    main()
