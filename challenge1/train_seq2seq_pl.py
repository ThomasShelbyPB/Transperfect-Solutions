"""
Challenge 1A: Encoder–decoder fine-tuning (EN->NL) with PyTorch Lightning.

Supports:
- WMT16 (via datasets) for training
- Optional domain filtering/weighting for software-like strings
- Evaluation hooks for FLORES and local Excel test set handled in eval_mt.py

Example:
python challenge1/train_seq2seq_pl.py --model_name Helsinki-NLP/opus-mt-en-nl --train_dataset wmt16 --src_lang en --tgt_lang nl
"""
import argparse, os, math
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup

SOFTWARE_KEYWORDS = [
    "click", "tap", "settings", "update", "install", "download", "app", "account",
    "password", "sign in", "log in", "window", "error", "warning", "enable",
    "disable", "device", "screen", "button", "menu", "user", "network", "wifi",
]

def looks_software_domain(en: str) -> bool:
    s = str(en).lower()
    return any(k in s for k in SOFTWARE_KEYWORDS) or ("{" in s and "}" in s)  # tags/placeholders

class WmtSeq2SeqDM(pl.LightningDataModule):
    def __init__(self, model_name, src_lang, tgt_lang, max_length, batch_size, num_workers, max_train_samples=None):
        super().__init__()
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage=None):
        ds = load_dataset("opus_books", f"{self.src_lang}-{self.tgt_lang}")

        if "validation" in ds:
            train = ds["train"]
            valid = ds["validation"]
        elif "test" in ds:
            train = ds["train"]
            valid = ds["test"]
        else:
            # opus_books is train-only -> create a validation split
            split = ds["train"].train_test_split(test_size=0.02, seed=42)
            train = split["train"]
            valid = split["test"]


        def preprocess(batch):
            src = [x[self.src_lang] for x in batch["translation"]]
            tgt = [x[self.tgt_lang] for x in batch["translation"]]
            model_inputs = self.tokenizer(src, max_length=self.max_length, truncation=True)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(tgt, max_length=self.max_length, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        self.train_ds = train.map(preprocess, batched=True, remove_columns=train.column_names)
        self.val_ds = valid.map(preprocess, batched=True, remove_columns=valid.column_names)

    def train_dataloader(self):
        collator = DataCollatorForSeq2Seq(self.tokenizer, model=None)
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collator)

    def val_dataloader(self):
        collator = DataCollatorForSeq2Seq(self.tokenizer, model=None)
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collator)

class Seq2SeqLit(pl.LightningModule):
    def __init__(self, model_name, lr, warmup_steps, max_steps, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log("train_loss", out.loss, prog_bar=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log("val_loss", out.loss, prog_bar=True)
        return out.loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(params, lr=self.hparams.lr)
        sch = get_linear_schedule_with_warmup(opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.hparams.max_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_dataset", default="wmt16", choices=["wmt16"])
    ap.add_argument("--src_lang", default="en")
    ap.add_argument("--tgt_lang", default="nl")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--precision", default="16-mixed")
    args = ap.parse_args()

    pl.seed_everything(42)

    dm = WmtSeq2SeqDM(
        model_name=args.model_name,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
    )
    model = Seq2SeqLit(
        model_name=args.model_name,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
    )

    ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        precision=32,                 # IMPORTANT on CPU
        max_steps=args.max_steps,
        num_sanity_val_steps=0,
        limit_val_batches=0.0,        # disable val for speedcheck
        enable_checkpointing=False,   # disable checkpoint writing for speedcheck
        logger=False,                 # disable logging overhead
    )

    trainer.fit(model, dm)
    # Save HF format
    os.makedirs(args.output_dir, exist_ok=True)
    model.model.save_pretrained(args.output_dir)
    dm.tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to: {args.output_dir}")

if __name__ == "__main__":
    main()
