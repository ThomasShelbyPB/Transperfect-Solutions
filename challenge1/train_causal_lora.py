"""
Challenge 1B: Decoder-only fine-tuning (EN->NL) with LoRA (PEFT).

We format each sample as:
  <BOS> Translate to Dutch: {source}\nDutch: {target} <EOS>

Train objective is next-token prediction over the full prompt+target; loss masking is supported
but kept minimal here for clarity.

Example:
python challenge1/train_causal_lora.py --model_name bigscience/bloom-560m --train_dataset wmt16 --src_lang en --tgt_lang nl --output_dir outputs/causal_bloom_lora
"""
import argparse, os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from peft import LoraConfig, get_peft_model, TaskType

PROMPT_TMPL = "Translate to Dutch: {src}\nDutch: "

def build_text(src: str, tgt: str) -> str:
    return PROMPT_TMPL.format(src=src) + str(tgt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_dataset", default="wmt16", choices=["wmt16"])
    ap.add_argument("--src_lang", default="en")
    ap.add_argument("--tgt_lang", default="nl")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_steps", type=int, default=15000)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    ds = load_dataset("wmt16", f"{args.src_lang}-{args.tgt_lang}")
    train = ds["train"]
    valid = ds["validation"]

    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))
        valid = valid.select(range(min(5000, len(valid))))

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else None)

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query_key_value", "dense", "fc_in", "fc_out"],  # BLOOM common modules; adjust per model
    )
    model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()

    def preprocess(batch):
        src = [x[args.src_lang] for x in batch["translation"]]
        tgt = [x[args.tgt_lang] for x in batch["translation"]]
        texts = [build_text(s, t) for s, t in zip(src, tgt)]
        out = tok(texts, max_length=args.max_length, truncation=True, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    train_tok = train.map(preprocess, batched=True, remove_columns=train.column_names)
    valid_tok = valid.map(preprocess, batched=True, remove_columns=valid.column_names)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=collator,
    )

    trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Saved LoRA fine-tuned model to: {args.output_dir}")

if __name__ == "__main__":
    main()
