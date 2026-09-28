#!/usr/bin/env python3
"""Fine-tune FinBERT end-to-end on the market question, against a matched baseline. ([16])

`pmv1_text_signal.py` freezes the encoder and reads it with a ridge. That is the
cheap test, and it says the question adds nothing to the price. The obvious
objection is that frozen mean-pooled features are a weak read of a transformer and
that fine-tuning would find what the ridge could not. This script removes the
objection by fine-tuning the whole encoder on the target.

The comparison is architecture-matched, which is the only way the answer means
anything. One head, one optimiser, one schedule, three arms:

  --arm price   MLP over the price block alone; the text tower is not built
  --arm text    the same MLP over [mean-pooled FinBERT ; price block], encoder
                unfrozen and trained jointly
  --arm null    identical to `text`, but the question attached to each market is
                permuted within (month x category x price decile) first

So `text - price` is what the language is worth under fine-tuning, and `null`
says how much of that a model this size manufactures from a destroyed corpus.
Reporting only the first difference would leave the second unmeasured, which is
the failure sec.4 documents.

Splits, weights, target and execution accounting are imported from the ridge
study so the two are directly comparable: one row per market-day, weight 1 per
market, y = win_event - p, and a book that longs the top quintile and shorts the
bottom at vwap_buy / vwap_sell, held to resolution.

Training rows are capped at --cap per market. A 470-day market otherwise supplies
470 nearly identical (question, label) pairs and the encoder memorises it; the cap
is the same instinct as the 1/rows weight in the ridge, applied where a weight
cannot reach.

    python pm/pmv1_text_finetune.py --arm text  --test-month 2026-02
    python pm/pmv1_text_finetune.py --arm price --test-month 2026-02
    python pm/pmv1_text_finetune.py --arm null  --test-month 2026-02
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pmv1_text_signal as S

NLP = S.NLP
REPO = "ProsusAI/finbert"


class Fusion(nn.Module):
    """Mean-pooled encoder output concatenated with the price block, then an MLP.

    The price block enters the head directly rather than through the encoder, so
    the text tower is never asked to re-derive a number it was handed, and the
    price arm is the same head with the text half absent.
    """

    def __init__(self, n_price, enc=None, hidden=256, dim=768):
        super().__init__()
        self.enc = enc
        d = n_price + (dim if enc is not None else 0)
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, xp, ids=None, mask=None):
        if self.enc is not None:
            h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            xp = torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1), xp], dim=-1)
        return self.head(xp).squeeze(-1)


def batches(n, bs, shuffle, gen=None):
    idx = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="text", choices=["text", "price", "null"])
    ap.add_argument("--field", default="text", choices=["text", "text_scrub"])
    ap.add_argument("--kind", default="nl")
    ap.add_argument("--test-month", default="2026-02")
    ap.add_argument("--cap", type=int, default=24, help="max training rows per market")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--maxlen", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = S.load(a.kind, "term")
    df = S.apply_null(df, "shuffle" if a.arm == "null" else "none", a.seed)

    t0 = pd.Timestamp(a.test_month + "-01")
    t1 = t0 + pd.offsets.MonthBegin(1)
    v0 = t0 - pd.Timedelta(days=S.VALID_DAYS)
    ef, dt = df.ev_first, df.date
    tr = ((ef < v0) & (dt < v0)).values
    va = ((ef >= v0) & (ef < t0) & (dt >= v0) & (dt < t0)).values
    te = ((ef >= t0) & (ef < t1) & (dt >= t0) & (dt < t1)).values

    # cap training rows per market -- see module docstring
    rng = np.random.default_rng(a.seed)
    tri = np.where(tr)[0]
    keep = (pd.Series(tri).groupby(df.market_id.values[tri])
            .apply(lambda s: s.sample(min(len(s), a.cap), random_state=a.seed))
            .explode().astype(int).values)
    tr_idx = np.sort(keep)

    cats = df.cat.value_counts().head(20).index.tolist()
    Xp = S.zscore(S.price_block(df, cats), tr)
    y, w = df.y.values, df.w.values
    print(f"arm={a.arm} test={a.test_month}  train {len(tr_idx):,} (capped from {tr.sum():,})"
          f"  valid {va.sum():,}  test {te.sum():,}  on {dev}")

    enc = tok = None
    if a.arm != "price":
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(REPO)
        enc = AutoModel.from_pretrained(REPO)
        E = tok(df[a.field].tolist(), padding="max_length", truncation=True,
                max_length=a.maxlen, return_tensors="pt")
        ids, mask = E["input_ids"], E["attention_mask"]

    model = Fusion(Xp.shape[1], enc).to(dev)
    groups = [{"params": model.head.parameters(), "lr": a.head_lr}]
    if enc is not None:
        groups.append({"params": model.enc.parameters(), "lr": a.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)

    XP = torch.tensor(Xp, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)
    W = torch.tensor(w, dtype=torch.float32)

    def predict(idx_np, bs=256):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx_np), bs):
                j = idx_np[i:i + bs]
                args = (XP[j].to(dev),)
                if enc is not None:
                    args += (ids[j].to(dev), mask[j].to(dev))
                out.append(model(*args).float().cpu().numpy())
        return np.concatenate(out)

    va_i, te_i = np.where(va)[0], np.where(te)[0]
    gen = torch.Generator().manual_seed(a.seed)
    best = (-9e9, None)
    hist = []
    for ep in range(a.epochs):
        model.train()
        tot = nb = 0
        t_start = time.time()
        for b in batches(len(tr_idx), a.bs, True, gen):
            j = tr_idx[b.numpy()]
            args = (XP[j].to(dev),)
            if enc is not None:
                args += (ids[j].to(dev), mask[j].to(dev))
            pred = model(*args)
            wt = W[j].to(dev)
            loss = (wt * (pred - Y[j].to(dev)) ** 2).sum() / wt.sum().clamp(min=1e-9)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
        v = S.ic(predict(va_i), y[va_i], df.event_id.values[va_i])["ic"]
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), valid_ic=v,
                         secs=time.time() - t_start))
        print(f"  epoch {ep}  loss {tot/max(nb,1):.5f}  valid IC {v:+.4f}  "
              f"({time.time()-t_start:.0f}s)")
        if v > best[0]:
            best = (v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})

    if best[1] is not None:
        model.load_state_dict(best[1])
    pt = predict(te_i)
    r = S.ic(pt, y[te_i], df.event_id.values[te_i])
    b = S.book_stats([S.book_rows(pt, df.iloc[te_i])])
    print(f"\ntest IC {r['ic']:+.4f} (t={r['ic_t']:+.2f})   best valid IC {best[0]:+.4f}")
    if b:
        print(f"  book: {b['trips']:,} trips / {b['n_events']:,} events   "
              f"gross {b['gross_pp']:+.4f} pp (t={b['t_gross']:+.2f})   "
              f"net {b['net_pp']:+.4f} pp (t={b['t_net']:+.2f})")

    res = dict(args=vars(a), history=hist, valid_ic=best[0], **r, book=b,
               n_train=int(len(tr_idx)), n_test=int(te.sum()))
    out = os.path.join(NLP, f"finetune_{a.arm}_{a.field}_{a.test_month}.json")
    json.dump(res, open(out, "w"), indent=2, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
