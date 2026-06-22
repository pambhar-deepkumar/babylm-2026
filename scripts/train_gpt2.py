"""Train a GPT-2 causal LM on a BabyLM corpus, matching the official 2026
Strict-Small baseline recipe.

Architecture and tokenizer are loaded from the official baseline repo so the
trained model is identical to the baseline in everything except the training
data; weights are initialised from scratch (`from_config`), not from the
baseline checkpoint. This is the "Arm B" run for the simplification study
(train data = the register-simplified corpus); pointing --train_file at the
original corpus reproduces the baseline ("Arm A").

Recipe (from the baseline config): GPT-2 12L/12H/768, ctx 1024, vocab 16384;
AdamW, lr 5e-4, cosine schedule, 20 epochs, effective batch 32768 tokens
(32 sequences x 1024), seed 42.

Example:
    python scripts/train_gpt2.py \
        --train_file data/bb26_simplified.train \
        --model_path baseline_gpt2_2026 \
        --output_dir output/arm_b_simplified \
        --epochs 20 --lr 5e-4 --per_device_batch 32 --seq_len 1024 --seed 42 --bf16
"""

from __future__ import annotations

import argparse
from itertools import chain

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

BASELINE = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True, help="raw text corpus, one document per line")
    p.add_argument("--valid_file", default=None, help="optional held-out text for eval loss")
    p.add_argument("--model_path", default=BASELINE,
                   help="HF id or local dir holding the baseline config.json + tokenizer")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--per_device_batch", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--num_proc", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_config(config)
    print(f"model: {model.num_parameters() / 1e6:.1f}M params | vocab {config.vocab_size} | ctx {args.seq_len}")

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = config.eos_token_id

    data_files = {"train": args.train_file}
    if args.valid_file:
        data_files["validation"] = args.valid_file
    raw = load_dataset("text", data_files=data_files)

    def tokenize(batch):
        ids = tokenizer(batch["text"], add_special_tokens=False)["input_ids"]
        # mark document boundaries with eos before packing
        return {"input_ids": [seq + [eos_id] for seq in ids]}

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"], num_proc=args.num_proc)

    def group(batch):
        concat = list(chain(*batch["input_ids"]))
        n = (len(concat) // args.seq_len) * args.seq_len
        blocks = [concat[i:i + args.seq_len] for i in range(0, n, args.seq_len)]
        return {"input_ids": blocks, "labels": [b[:] for b in blocks]}

    lm = tokenized.map(group, batched=True, num_proc=args.num_proc)
    print(f"train blocks: {len(lm['train'])} x {args.seq_len} tokens")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        seed=args.seed,
        bf16=args.bf16,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch" if args.valid_file else "no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=lm["train"],
        eval_dataset=lm.get("validation"),
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"done -> {args.output_dir}")


if __name__ == "__main__":
    main()
