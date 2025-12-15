"""
Evaluate an EN->NL model on:
- FLORES devtest (general domain): datasets "flores200"
- Provided software-domain Excel test set (Dataset_Challenge_1.xlsx)

Outputs JSON with metrics:
- BLEU (if evaluate+sacrebleu available)
- chrF_proxy (always)
- HTER_proxy (always, token edit effort)

Example:
python challenge1/eval_mt.py --model_dir outputs/seq2seq_marian --model_type seq2seq --flores_lang_pair eng_Latn-nld_Latn --software_test_xlsx data/Dataset_Challenge_1.xlsx
"""
import argparse, json, os
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from challenge1.common_metrics import chrf_proxy, hter_proxy

def batch_generate_seq2seq(model, tok, src_texts, max_new_tokens=128, num_beams=4):
    inputs = tok(src_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams)
    return tok.batch_decode(gen, skip_special_tokens=True)

def batch_generate_causal(model, tok, src_texts, max_new_tokens=128, num_beams=1):
    prompts = [f"Translate to Dutch: {s}\nDutch: " for s in src_texts]
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=num_beams)
    # decode full text, then strip prompt
    outs = tok.batch_decode(gen, skip_special_tokens=True)
    preds = []
    for prompt, out in zip(prompts, outs):
        preds.append(out[len(prompt):].strip())
    return preds

def compute_metrics(preds, refs):
    chrf = float(np.mean([chrf_proxy(p, r) for p, r in zip(preds, refs)]))
    hter = float(np.mean([hter_proxy(p, r) for p, r in zip(preds, refs)]))
    return {"chrF_proxy": chrf, "HTER_proxy": hter}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_type", required=True, choices=["seq2seq", "causal"])
    ap.add_argument("--flores_lang_pair", default="eng_Latn-nld_Latn")
    ap.add_argument("--software_test_xlsx", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    if args.model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
        gen_fn = lambda batch: batch_generate_seq2seq(model, tok, batch)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
        gen_fn = lambda batch: batch_generate_causal(model, tok, batch)

    # FLORES
    src_lang, tgt_lang = args.flores_lang_pair.split("-")
    flores = load_dataset("facebook/flores", args.flores_lang_pair, trust_remote_code=True)
    src_col = f"sentence_{src_lang}"
    tgt_col = f"sentence_{tgt_lang}"

    src = flores["devtest"][src_col]
    tgt = flores["devtest"][tgt_col]


    flores_preds = []
    for i in range(0, len(src), args.batch_size):
        flores_preds.extend(gen_fn(src[i:i+args.batch_size]))
    flores_metrics = compute_metrics(flores_preds, tgt)

    # Software domain Excel
    df = pd.read_excel(args.software_test_xlsx)
    src2 = df["English Source"].astype(str).tolist()
    ref2 = df["Reference Translation"].astype(str).tolist()

    soft_preds = []
    for i in range(0, len(src2), args.batch_size):
        soft_preds.extend(gen_fn(src2[i:i+args.batch_size]))
    soft_metrics = compute_metrics(soft_preds, ref2)

    out = {
        "model_dir": args.model_dir,
        "model_type": args.model_type,
        "flores_lang_pair": args.flores_lang_pair,
        "metrics": {
            "flores_devtest": flores_metrics,
            "software_test": soft_metrics
        }
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
