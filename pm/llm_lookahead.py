#!/usr/bin/env python3
"""Does a language model forecast a prediction market, or remember it?

RESULTS.md sec.3.6 asks whether an *encoder* can read a Polymarket question, and
finds the text is a clock. The question a reader asks next is about generative
LLMs, and there the leak is sharper: a model trained on the web after a market
resolved may simply know the answer. That is look-ahead through the pretraining
corpus, and nothing in a backtest announces it.

Design. Each market is asked once, from its question text alone, and the model's
P(Yes) is read off the next-token distribution -- no sampling, no chain of
thought, so there is nothing to tune and one number per (model, market):

    p_llm = P("Yes") / (P("Yes") + P("No"))    over the assistant turn's first token

Every model sees the same 12.6k markets (data/llm/markets.parquet: first
uncertain market-day, 0.03 < p < 0.97, >=1 day to resolution). The models are
chosen so their *documented* training cutoffs fall at different points inside
the markets' resolution window. If the model is forecasting, its edge over the
market price should not care where a market resolved relative to its cutoff. If
it is remembering, the edge should be large before the cutoff and vanish after,
and -- the part no single model can show -- the same market should look
predictable to a later-cutoff model and not to an earlier one.

    python pm/llm_lookahead.py --model Qwen/Qwen3.5-9B
    python pm/llm_lookahead.py --model google/gemma-4-12B-it --quant 8bit
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
LLM = os.path.join(ROOT, "data", "llm")

# Two prompts, both fixed in advance and both reported. `forecast` asks the question
# as a market does, which is the honest use; but it frames the event as future, and
# an instruct model may then decline to use what it knows. `recall` asks what
# actually happened, so it is the upper bound on what the pretraining corpus leaks.
PROMPTS = {
    "forecast": ("The following is a question from a prediction market.\n\n"
                 "Question: {q}?\n\n"
                 "Will this question resolve YES? Answer with a single word, Yes or No."),
    "recall": ("The following is a question from a prediction market that has since "
               "resolved.\n\nQuestion: {q}?\n\n"
               "Based on what you know about what actually happened, did it resolve YES? "
               "Answer with a single word, Yes or No."),
}


def answer_ids(tok):
    """Token ids that begin a Yes / No answer, across the spellings tokenizers use."""
    out = {}
    for lab, words in (("yes", ["Yes", " Yes", "yes", " yes", "YES"]),
                       ("no", ["No", " No", "no", " no", "NO"])):
        ids = set()
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if t:
                ids.add(t[0])
        out[lab] = sorted(ids)
    assert not set(out["yes"]) & set(out["no"]), out
    return out


def build_prompt(tok, q, prompt="forecast"):
    msgs = [{"role": "user", "content": PROMPTS[prompt].format(q=q)}]
    kw = dict(tokenize=False, add_generation_prompt=True)
    try:                                   # thinking models: answer directly
        return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", choices=list(PROMPTS), default="forecast")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(LLM, "p_" + a.model.replace("/", "__")
                                + ("" if a.prompt == "forecast" else f"__{a.prompt}") + ".parquet")
    if os.path.exists(out):
        print("exists", out); return

    m = pd.read_parquet(os.path.join(LLM, "markets.parquet"))
    if a.limit:
        m = m.head(a.limit)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
    if a.quant != "none":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (BitsAndBytesConfig(load_in_8bit=True) if a.quant == "8bit"
                                     else BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_compute_dtype=torch.bfloat16,
                                          bnb_4bit_quant_type="nf4"))
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw).eval()
    except ValueError:
        # multimodal checkpoints (Mistral3, some Gemma/Qwen releases) register only
        # an image-text class; fed text alone it is the same language model
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(a.model, **kw).eval()
    print(f"loaded {a.model} in {time.time()-t0:.0f}s, "
          f"{torch.cuda.memory_allocated()/2**30:.1f} GB", flush=True)
    ids = answer_ids(tok)
    print("answer ids", {k: [tok.decode([i]) for i in v] for k, v in ids.items()}, flush=True)
    prompts = [build_prompt(tok, q, a.prompt) for q in m.text]
    print("example prompt:\n" + prompts[0], flush=True)

    p_yes, mass = [], []
    t0 = time.time()
    for i in range(0, len(prompts), a.batch):
        enc = tok(prompts[i:i + a.batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            # only the last position is read; materialising logits for every position
            # of a 262k-token vocabulary (Gemma) exhausts 24 GB at batch 32
            try:
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            except TypeError:
                logits = model(**enc).logits[:, -1, :].float()
        pr = torch.softmax(logits, -1)
        y = pr[:, ids["yes"]].sum(1); n = pr[:, ids["no"]].sum(1)
        p_yes += (y / (y + n)).cpu().tolist(); mass += (y + n).cpu().tolist()
        if (i // a.batch) % 50 == 0:
            print(f"  {i+len(enc.input_ids)}/{len(prompts)} {time.time()-t0:.0f}s", flush=True)
    res = pd.DataFrame({"market_id": m.market_id.values, "p_llm": p_yes, "answer_mass": mass})
    res["model"] = a.model
    res["prompt"] = a.prompt
    # log-odds straight from the two masses, so no clip is ever needed downstream
    res["logodds"] = [float(np.log(max(p, 1e-30)) - np.log(max(1 - p, 1e-30))) for p in p_yes]
    res.to_parquet(out)
    print(f"wrote {out}: mean p_llm {np.mean(p_yes):.3f}, median answer mass "
          f"{np.median(mass):.3f}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
