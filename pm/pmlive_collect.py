#!/usr/bin/env python3
"""Capture the Polymarket CLOB book as minute bars into data/pmlive/.

The raw websocket firehose is ~34 GB/day for this universe, which is 3 TB over a
three-month run.  Aggregating to one-minute bars in memory and writing only the
bars costs ~11.5 B/row, i.e. ~50-100 MB/day -- a ~150x reduction for data that is
still minute-resolution.  Nothing raw is kept, so the aggregation is the archive:
it records mid OHLC, both touches with their sizes, and 5-level depth notional.

THE TRAP THAT MATTERS.  The market websocket silently caps a subscription
somewhere between 700 and 800 asset ids.  At 700 you get 698 book snapshots; at
800 you get *ten*, with no error and a connection that looks healthy.  A single
subscription for the whole universe records ~1% of the market while appearing to
work.  Hence SHARD=450 across concurrent connections, an assert, and a coverage
figure logged every minute -- if coverage falls, the capture is lying to you.

Connections also close spontaneously (observed at 550 ids), so every shard
reconnects with backoff and re-subscribes; a fresh `book` snapshot rebuilds that
shard's ladders, so a reconnect self-heals rather than drifting.
"""
import argparse, asyncio, collections, datetime as dt, json, os, signal, sqlite3, sys, time
import pyarrow as pa, pyarrow.parquet as pq
import websockets

URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, os.pardir, "data", "pmlive.sqlite")
OUT = os.path.join(HERE, os.pardir, "data", "pmlive")
SHARD = 450               # hard-verified safe; the cliff is between 700 and 800
MAXSUB = 600              # refuse anything near the cliff
FLUSH_SEC = 300           # cap in-memory exposure to 5 minutes of bars

BAR = pa.schema([("tok", pa.int32()), ("ts", pa.int32()),
                 ("o", pa.int16()), ("h", pa.int16()), ("l", pa.int16()), ("c", pa.int16()),
                 ("bb", pa.int16()), ("ba", pa.int16()),
                 ("bsz", pa.float32()), ("asz", pa.float32()),
                 ("bd5", pa.float32()), ("ad5", pa.float32()), ("n", pa.int16())])
TRD = pa.schema([("tok", pa.int32()), ("ts", pa.int32()), ("price", pa.int16()),
                 ("size", pa.float32()), ("side", pa.int8())])


def load_universe():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = c.execute("SELECT token_id, tok FROM tokens WHERE tracking=1").fetchall()
    c.close()
    return {t: i for t, i in rows}


class Collector:
    def __init__(self, uni, dense=True):
        self.idx = uni                      # token_id -> tok
        self.dense = dense
        self.book = {}                      # token_id -> (bids{px:sz}, asks{px:sz})
        self.bars = {}                      # (tok, minute) -> list
        self.trades = []
        self.dirty = set()
        self.msgs = self.drops = 0
        self.shards_up = 0
        self.buf_bar, self.buf_trd = [], []
        self.cur_hour = None
        self.last_write = time.time()

    # ---- book maths -------------------------------------------------------
    @staticmethod
    def _px(s):
        try:
            return int(round(float(s) * 1000))
        except Exception:
            return 0

    def snap(self, tid):
        b, a = self.book.get(tid, ({}, {}))
        bb = max(b, key=float, default=None)
        ba = min(a, key=float, default=None)
        bbv = self._px(bb) if bb else 0
        bav = self._px(ba) if ba else 0
        bsz = float(b[bb]) if bb else 0.0
        asz = float(a[ba]) if ba else 0.0
        bd5 = sum(float(b[p]) for p in sorted(b, key=float, reverse=True)[:5])
        ad5 = sum(float(a[p]) for p in sorted(a, key=float)[:5])
        mid = (bbv + bav) // 2 if (bbv and bav) else (bbv or bav)
        return mid, bbv, bav, bsz, asz, bd5, ad5

    def touch(self, tid, minute):
        tok = self.idx.get(tid)
        if tok is None:
            return
        mid, bb, ba, bsz, asz, bd5, ad5 = self.snap(tid)
        k = (tok, minute)
        r = self.bars.get(k)
        if r is None:
            self.bars[k] = [mid, mid, mid, mid, bb, ba, bsz, asz, bd5, ad5, 1]
        else:
            r[3] = mid
            if mid > r[1]:
                r[1] = mid
            if mid < r[2] or r[2] == 0:
                r[2] = mid
            r[4], r[5], r[6], r[7], r[8], r[9] = bb, ba, bsz, asz, bd5, ad5
            r[10] += 1
        self.dirty.add(tid)

    # ---- message handling -------------------------------------------------
    def handle(self, x, minute):
        et = x.get("event_type")
        if et == "book":
            tid = x.get("asset_id")
            if not tid:
                return
            self.book[tid] = ({e["price"]: e["size"] for e in (x.get("bids") or [])},
                              {e["price"]: e["size"] for e in (x.get("asks") or [])})
            self.touch(tid, minute)
        elif et == "price_change":
            # NOTE: no top-level asset_id. Each entry of price_changes[] names its
            # own asset, so one message can span several tokens.
            for c in (x.get("price_changes") or []):
                tid = c.get("asset_id")
                p = c.get("price")
                if not tid or p is None:
                    continue
                b, a = self.book.setdefault(tid, ({}, {}))
                tgt = b if (c.get("side") or "").upper() == "BUY" else a
                try:
                    zero = float(c.get("size") or 0) == 0
                except Exception:
                    zero = True
                if zero:
                    tgt.pop(p, None)
                else:
                    tgt[p] = c["size"]
                self.touch(tid, minute)
        elif et == "last_trade_price":
            tid = x.get("asset_id")
            tok = self.idx.get(tid)
            if tok is None:
                return
            try:
                self.trades.append((tok, int(int(x["timestamp"]) / 1000),
                                    self._px(x.get("price")), float(x.get("size") or 0),
                                    1 if (x.get("side") or "").upper() == "BUY" else 0))
            except Exception:
                pass

    # ---- shard ------------------------------------------------------------
    async def shard(self, sub, stop_evt, name):
        assert len(sub) <= MAXSUB, f"shard {len(sub)} exceeds safe cap {MAXSUB}"
        backoff = 1
        while not stop_evt.is_set():
            try:
                async with websockets.connect(URL, open_timeout=30, ping_interval=10,
                                              ping_timeout=25, max_size=2 ** 25) as ws:
                    await ws.send(json.dumps({"assets_ids": sub, "type": "market"}))
                    self.shards_up += 1
                    backoff = 1
                    try:
                        while not stop_evt.is_set():
                            m = await asyncio.wait_for(ws.recv(), timeout=90)
                            self.msgs += 1
                            minute = int(time.time() // 60)
                            d = json.loads(m)
                            for x in (d if isinstance(d, list) else [d]):
                                self.handle(x, minute)
                    finally:
                        self.shards_up -= 1
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.drops += 1
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    # ---- output -----------------------------------------------------------
    def _paths(self, when):
        d = when.strftime("%Y-%m-%d")
        h = when.strftime("%H")
        bd = os.path.join(OUT, "bars", f"dt={d}")
        td = os.path.join(OUT, "trades", f"dt={d}")
        os.makedirs(bd, exist_ok=True)
        os.makedirs(td, exist_ok=True)
        return os.path.join(bd, f"bars-{h}.parquet"), os.path.join(td, f"trades-{h}.parquet")

    def _write(self, when):
        """Flush buffered rows to this hour's parquet. Sorted for compression."""
        bp, tp = self._paths(when)
        if self.buf_bar:
            t = pa.Table.from_arrays(
                [pa.array(c, ty) for c, ty in zip(zip(*self.buf_bar), BAR.types)],
                schema=BAR).sort_by([("tok", "ascending"), ("ts", "ascending")])
            self._append(bp, t)
            self.buf_bar = []
        if self.buf_trd:
            t = pa.Table.from_arrays(
                [pa.array(c, ty) for c, ty in zip(zip(*self.buf_trd), TRD.types)],
                schema=TRD).sort_by([("tok", "ascending"), ("ts", "ascending")])
            self._append(tp, t)
            self.buf_trd = []

    @staticmethod
    def _append(path, t):
        if os.path.exists(path):
            t = pa.concat_tables([pq.read_table(path), t]).sort_by(
                [("tok", "ascending"), ("ts", "ascending")])
        tmp = path + ".tmp"
        pq.write_table(t, tmp, compression="zstd", compression_level=9,
                       use_dictionary=["tok"])
        os.replace(tmp, path)

    def flush(self, upto_minute, n_sub, log):
        """Move completed minutes into the hour buffer; write on hour rollover.

        Dense mode emits one row per token that has a book, every minute: a token
        with no update simply repeats its standing quote (n=0).  That keeps the
        panel rectangular so downstream work needs no forward-fill, and it costs
        little because repeated rows compress almost to nothing.
        """
        done = sorted(k for k in self.bars if k[1] < upto_minute)
        emitted = set()
        for k in done:
            r = self.bars.pop(k)
            emitted.add(k[0])
            self.buf_bar.append((k[0], k[1], r[0], r[1], r[2], r[3], r[4], r[5],
                                 r[6], r[7], r[8], r[9], r[10]))
        if self.dense and done:
            minute = upto_minute - 1
            for tid, tok in self.idx.items():
                if tok in emitted:
                    continue
                b, a = self.book.get(tid, ({}, {}))
                if not b and not a:
                    continue                      # never had a book: no row
                mid, bb, ba, bsz, asz, bd5, ad5 = self.snap(tid)
                self.buf_bar.append((tok, minute, mid, mid, mid, mid, bb, ba,
                                     bsz, asz, bd5, ad5, 0))
        self.buf_trd.extend(self.trades)
        self.trades = []
        now = dt.datetime.now(dt.UTC)
        hour = now.strftime("%Y%m%d%H")
        if self.cur_hour is None:
            self.cur_hour = hour
        if hour != self.cur_hour:
            self._write(now - dt.timedelta(hours=1))
            self.cur_hour = hour
            self.last_write = time.time()
        elif (len(self.buf_bar) > 400_000
              or time.time() - self.last_write > FLUSH_SEC):
            # Bound what an unclean kill can lose. Without this the hour's bars
            # live only in memory until rollover, so a SIGKILL or power cut costs
            # up to 60 minutes of capture that cannot be backfilled.
            self._write(now)
            self.last_write = time.time()
        cov = sum(1 for t in self.idx if self.book.get(t, ({}, {}))[0]
                  or self.book.get(t, ({}, {}))[1])
        line = (f"{now:%F %T} shards={self.shards_up} sub={n_sub} books={cov} "
                f"cov={cov / max(1, n_sub) * 100:.1f}% msgs={self.msgs} "
                f"drops={self.drops} bars_buf={len(self.buf_bar)}")
        print(line, flush=True)
        log.write(line + "\n"); log.flush()
        if cov < 0.9 * n_sub:
            warn = (f"  !! coverage {cov}/{n_sub} -- subscription cap or dead shard; "
                    f"check shard size <= {MAXSUB}")
            print(warn, flush=True); log.write(warn + "\n"); log.flush()
        self.msgs = 0


async def run(args):
    os.makedirs(OUT, exist_ok=True)
    uni = load_universe()
    if not uni:
        sys.exit("universe empty -- run pm/pmlive_universe.py first")
    col = Collector(uni, dense=not args.sparse)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, stop.set)

    toks = list(uni)
    chunks = [toks[i:i + args.shard] for i in range(0, len(toks), args.shard)]
    print(f"universe {len(toks):,} tokens -> {len(chunks)} shards of <= {args.shard}")
    tasks = [asyncio.create_task(col.shard(c, stop, f"s{i}")) for i, c in enumerate(chunks)]

    log = open(os.path.join(OUT, "health.log"), "a")
    deadline = time.time() + args.minutes * 60 if args.minutes else None
    last = int(time.time() // 60)
    try:
        while not stop.is_set():
            await asyncio.sleep(2)
            m = int(time.time() // 60)
            if m != last:
                col.flush(m, len(toks), log)
                last = m
            if deadline and time.time() > deadline:
                break
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        col.flush(int(time.time() // 60) + 1, len(toks), log)
        col._write(dt.datetime.now(dt.UTC))
        log.close()
        print("stopped cleanly; buffers flushed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=SHARD)
    p.add_argument("--minutes", type=int, default=0, help="0 = run until signalled")
    p.add_argument("--sparse", action="store_true",
                   help="only write tokens that updated in the minute")
    a = p.parse_args()
    if a.shard > MAXSUB:
        sys.exit(f"--shard {a.shard} is past the verified-safe cap {MAXSUB}")
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
