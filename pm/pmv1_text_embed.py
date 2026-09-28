#!/usr/bin/env python3
"""Cache frozen sentence embeddings for every distinct market question. ([16])

Three encoders, chosen so the comparison answers a question rather than just
producing a number:

  ProsusAI/finbert          BERT further-pretrained on financial news, then tuned
                            for sentiment. The domain-specific candidate.
  bert-base-uncased         its own base model. Isolates what the financial
                            pretraining actually bought.
  all-MiniLM-L6-v2          a general sentence encoder trained for semantic
                            similarity rather than for a classification head.

Embeddings are mean-pooled over the attention mask, not the CLS vector: FinBERT's
CLS is shaped by its sentiment head and is a poor frozen feature, while mean
pooling is what MiniLM was trained to produce. Both text variants from the panel
are encoded -- `text` verbatim and `text_scrub` with clock readings masked -- so
the downstream regressions can separate language from calendar.

The vocabulary is small (31,877 markets), so this is a couple of minutes on one
GPU and the cache is reused by every arm of the study.

    python pm/pmv1_text_embed.py --model finbert --field text
"""
import argparse, os, sys
import duckdb, numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
NLP = os.path.join(HERE, os.pardir, "data", "nlp")
MODELS = {"finbert": "ProsusAI/finbert",
          "bert": "bert-base-uncased",
          "minilm": "sentence-transformers/all-MiniLM-L6-v2"}


def embed(texts, repo, device, batch=256, maxlen=64):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(repo)
    mod = AutoModel.from_pretrained(repo).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True,
                    max_length=maxlen, return_tensors="pt").to(device)
            h = mod(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
            if i % (batch * 20) == 0:
                print(f"  {i}/{len(texts)}", flush=True)
    return np.vstack(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="finbert", choices=list(MODELS))
    ap.add_argument("--field", default="text", choices=["text", "text_scrub"])
    ap.add_argument("--panel", default=os.path.join(NLP, "panel.parquet"))
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = duckdb.sql(f"""SELECT DISTINCT market_id, {a.field} AS txt
                        FROM read_parquet('{a.panel}')""").df()
    df = df.sort_values("market_id").reset_index(drop=True)
    print(f"{a.model} / {a.field}: {len(df)} markets on {dev}")

    E = embed(df.txt.tolist(), MODELS[a.model], dev)
    out = os.path.join(NLP, f"emb_{a.model}_{a.field}.npz")
    np.savez_compressed(out, market_id=df.market_id.values, emb=E)
    print(f"wrote {out}  shape={E.shape}")


if __name__ == "__main__":
    main()
