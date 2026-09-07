# Pulling data from findata

How `data/findata.sqlite` was built and how to rebuild or extend it. Source:
[lum.id findata](https://lum.id/findata) — REST + Postgres + streaming financial data.
Distilled from the session transcript in `data.md` plus the scripts that grew out of it.

## 1. Auth

Every data route needs a bearer token; `/`, `/reference`, `/openapi.json`, `/status`,
`/health`, `/freshness`, `/docs` are public.

```bash
set -a; source .env; set +a     # exports LUMID_TOKEN=lm_pat_live_…
curl -s -H "Authorization: Bearer $LUMID_TOKEN" https://lum.id/findata/usage/me
```

`.env` is git-ignored. Every script below reads `LUMID_TOKEN` from the environment and
exits if it is missing. Rate limit is 6000/min, surfaced as `x-ratelimit-limit` /
`-remaining` / `-reset`; 429 carries `Retry-After`. Eight workers runs ~90–96 req/s and
has never tripped it.

Before a big pull, sanity-check the backend: `/status` can report `ALL OK` while every
DB-backed route 500s. The honest signal is `/freshness` — an all-zero board (`0 green
0 amber 0 red`) means the data plane is down, wait it out; real SLA counts mean it is
serving.

## 2. The four access paths

| Path | Use for |
|---|---|
| REST `https://lum.id/findata/…` | everything here; ~169 routes |
| Postgres wire `sql.lum.id:5432` db `findata` user `lumid_reader` | bulk history in 2 queries instead of 23k requests — **but we have no password and no psql/duckdb/psycopg installed**, so unused |
| SSE / WebSocket `/quotes/stream`, `wss://lum.id/findata/ws/quotes` | live ticks; `/prediction-markets/stream` is broken |
| MCP `POST /mcp` | 92 auto-generated tools, one per read endpoint |

Machine-readable spec: `https://lum.id/findata/openapi.json` — generate against it rather
than hand-rolling routes.

## 3. Endpoints that matter

- **Universe** — `/universe` is a *historical* roster of 7,851 US symbols including dead
  tickers; `/universe/actively-trading?limit=100000` is a 69,946-symbol live global list.
  Join against the latter to drop delisted names; keep them in to avoid survivorship bias.
- **Prices** — `/ohlc/{sym}?interval=1d&from=&to=`. Intervals are exactly
  `1min|5min|15min|30min|1hour|4hour|1d` (`daily` is invalid). `limit` is ignored — narrow
  the date window instead. Serves back to ~2006, not 2 years.
- **Corporate actions** — `/dividends/{sym}?limit=5000`, `/splits/{sym}`.
- **Fundamentals** — `/fundamentals/{sym}/latest`, and
  `/fundamentals/{sym}/history?statement=income|balance|cashflow&period=quarter&limit=80`.
  `statement` is required; the default limit silently truncates.
- **Metadata** — `/symbols?ticker=A,B,…` in batches of 500.
- Other useful families: `/ratios/`, `/key-metrics/`, `/dcf/`, `/financial-scores/`
  (Altman Z, Piotroski F), `/news/{sym}`, `/kols/tweets/search`, `/prediction-markets/…`.

Conventions: dates take RFC3339 or bare `YYYY-MM-DD`; `from`/`to` ≡ `start`/`end` ≡
`since`/`until`; lists are comma-separated; ETag + `If-None-Match` gives 304s; most list
routes cap at 1000 rows — **page by time, never by offset**.

## 4. The scripts

All resumable: each logs `(symbol, kind) → status` and re-running skips anything already
logged 200/404/400, so a stall or a kill costs nothing. All write to
`data/findata.sqlite` (WAL, `synchronous=NORMAL`, `busy_timeout` set).

```bash
set -a; source .env; set +a

python3 scripts/findata_pull.py  --workers 8   # pass 1: 2y daily bars + fundamentals
python3 scripts/findata_meta.py                # symbol metadata + actively-trading list
python3 scripts/findata_pull2.py --workers 8   # pass 2: deep history, divs, splits, bal/cf
python3 scripts/pull_income.py                 # pass 3: refill income stmt at limit=80
```

Run them in that order on a cold start (`pull2` seeds its symbol list from
`ohlc_daily`, `pull_income` from `ohlc`). Add `--limit N` to smoke-test on N symbols
first — always worth it. Pass 1 is ~23.5k requests in ~4 minutes.

Why three passes: pass 1 took only 2 years and the income statement; pass 2 backfilled
what that left out (history to 2006, dividends, splits, balance sheet, cash flow, plus
12 benchmark tickers); pass 3 refilled income at `limit=80` because `limit=12` capped
point-in-time backtests at ~3 years and silently emptied the universe at earlier
formation dates.

## 5. What is in the DB

`data/findata.sqlite` — 2.2 GB, git-ignored (re-pullable).

| Table | Rows | Coverage |
|---|---:|---|
| `ohlc` | 15,207,829 | daily bars 2006-01-03 → 2026-08-14, 4,640 symbols |
| `ohlc_daily` | 2,012,954 | pass-1 2y window, 4,628 symbols (superseded by `ohlc`) |
| `fundamentals_history` | 270,963 | quarterly income, 6,211 symbols, back to 1968 |
| `fund_balance` | 248,169 | quarterly balance sheet, 4,502 symbols |
| `fund_cashflow` | 249,247 | quarterly cash flow |
| `fundamentals_latest` | 6,270 | latest quarter, 25 wide columns |
| `dividends` | 256,638 | 2,870 symbols (future-dated rows exist) |
| `splits` | 7,070 | 1,965 symbols |
| `symbols` | 7,850 | name, exchange, sector, industry, market cap, IPO date |
| `actively_trading` | 69,946 | live-symbol reference list |
| `fetch_log` / `fetch_log2` / `fetch_log3` | 23,553 / 23,200 / 4,640 | per-request audit trail, one per pass |

Fundamentals are stored as JSON `payload` blobs, not exploded into columns.

## 6. Traps — read before analysing

- **`close` is already a total-return series** (split *and* dividend adjusted), even
  though `adj_close` and `vwap` are NULL for every row. The NULLs read like "unadjusted"
  and are a red herring. Do **not** apply `dividends` on top — that double-counts and
  produced a fake 30%/yr yield for NVDA.
- **`dividends.amount`** is cash as paid, **`adj_amount`** is split-adjusted; they differ
  by up to 200x.
- **ETFs are useless here.** SPY returns ~30 bars/year and `/dividends/SPY` is `[]`.
  Build benchmarks out of individual stocks.
- **`symbols.market_cap` is a pull-time snapshot.** Derive shares against each symbol's own
  last traded price; a `notna()` filter on it deletes every delisted name and reintroduces
  survivorship bias.
- **3,223 of 7,851 universe symbols return zero bars** — delisted tickers (AABA, AAI,
  AACC…), none present in `actively_trading`. 404 is terminal and is logged as such so
  resumes do not re-request them.
- **1,574 symbols have no fundamentals** (mostly ETFs and dead names) — legitimately 404.
- **8 malformed universe entries** (`CFX 5.75`, `CHNG 6`, `DTE 6.25`, `IFF 6`) are
  bond/preferred tickers the API rejects as 400. Not worth chasing.
- **Weekend quotes** come back `"source":"prev_close","stale":true` — expected, not a fault.
- **Prediction markets** have their own hard limits (tape starts 2026-05-21, Polymarket LOB
  capture stopped, `/prediction-markets/stream` broken). See the notes in
  `fyp_ideas.md` / session memory before building on them.

## 7. Verifying a pull

```bash
python3 - <<'EOF'
import sqlite3
c = sqlite3.connect("data/findata.sqlite")
print(c.execute("SELECT MIN(ts), MAX(ts), COUNT(DISTINCT symbol), COUNT(*) FROM ohlc").fetchone())
for row in c.execute("SELECT kind, status, COUNT(*) FROM fetch_log2 GROUP BY kind, status ORDER BY kind, 3 DESC"):
    print(row)
EOF
```

A healthy pass-1 window shows a median of ~494 bars per live symbol over two years —
exactly the US trading-day count. Anything far below that means truncation, not a market.

## 8. Housekeeping

The token in `.env` was pasted into a chat transcript at some point (`data.md`), so
rotating it is cheap insurance. Keep it in the env var; never inline it into a script.

## 9. Off-vendor data: the Polymarket-v1 archive

Everything above is findata. One dataset used in `RESULTS.md` §3 is not, and it needs no
token at all.

**`TimeSeventeen/Polymarket-v1`** on Hugging Face (CC-BY-4.0, ungated, arXiv:2606.04217) is
an on-chain archive of Polymarket's first-generation CTF exchange on Polygon,
**2022-11-21 → 2026-04-28**. It matters here because the findata prediction-market tape
starts 2026-05-21, carries no aggressor field, and lost its outcome route (§6). The archive
predates and dwarfs it, and supplies the taker side from blockchain settlement rather than
from a classifier.

Four layers. Only the two cleaned trade layers are pulled:

| layer | size | what it is |
|---|---:|---|
| `daily_aligned/` | 13.2 GB | standard binary markets, relayers filtered, metadata joined |
| `daily_aligned_multi/` | 3.6 GB | negative-risk multi-outcome, same pipeline + `neg_risk_market_id` |
| `OrderFilled/` | 27.4 GB | raw nominal tape, 1.20 B rows, relayers **not** filtered |
| `CTF/` | 8.5 GB | contract lifecycle: preparations, splits, merges, resolutions, redemptions |

```bash
pip install duckdb pyarrow huggingface_hub
python pm/pmv1_pull.py               # 2,105 files, 16.8 GB, ~13 min
python pm/pmv1_prep.py               # -> data/pmv1_trades.parquet, data/pmv1_bars_1h.parquet
```

### Traps

- **`price` is not the event probability.** Use `p_event` (both legs mapped onto the
  `outcome_seq = 1` axis) and `D ∈ {+1,−1}` for direction. A `No`-token row at 0.56 is
  `p_event = 0.44`.
- **`block_timestamp` is seconds, not milliseconds.**
- **Do not pool the two cleaned layers without checking `neg_risk`.** Standard binary and
  negative-risk multi-outcome are structurally different; `p_event` in the multi layer is
  per candidate and is *not* designed to sum to 1 across a `neg_risk_market_id`.
- **Do not compute volume from `OrderFilled/`** without excluding the two relayer
  addresses listed in the dataset card. The cleaned layers already have.
- **`fee_usdc` is identically zero** across the archive — Polymarket v1 charged no explicit
  taker fee — and `taker_base_fee` is non-zero on only ~3.6 % of fills. Trading cost on this
  venue is the spread and must be estimated, not read off a column.
- **The relayer filter drops relayer *takers*,** so aggregate taker statistics are computed
  over a filtered side of the book.
- **Never load it into memory.** 746 M rows across the two cleaned layers. Use DuckDB or a
  lazy columnar scan; `pm/pmv1_prep.py` projects it once into two narrow tables so
  downstream passes are cheap (the first attempt at an account group-by straight off the
  wide layers spilled 33 GB to disk).

### Verifying a pull

```bash
python3 - <<'EOF'
import duckdb
print(duckdb.sql("""
  SELECT count(*) AS fills, count(DISTINCT taker_id) AS accounts,
         count(DISTINCT event_id) AS events, min(ts) AS t0, max(ts) AS t1
  FROM read_parquet('data/pmv1_trades.parquet')""").df())
EOF
```

A healthy pull shows **730,492,126** resolved fills, **2,588,367** accounts and
**704,521** events spanning 2022-11-21 to 2026-04-28.
