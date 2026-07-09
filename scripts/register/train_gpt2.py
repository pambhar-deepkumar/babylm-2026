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
    python scripts/register/train_gpt2.py \
        --train_file data/bb26_simplified.train \
        --model_path baseline_gpt2_2026 \
        --output_dir output/arm_b_simplified \
        --epochs 20 --lr 5e-4 --per_device_batch 32 --seq_len 1024 --seed 42 --bf16
"""

from __future__ import annotations

import argparse
import os
from itertools import chain

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

BASELINE = "BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small"

# Learning-trajectory checkpoints, expressed as *tokens seen* (millions). These
# mirror the official Arm A baseline's chck_1M..chck_100M revisions and extend a
# few points into the tail of the 20-epoch run, so an A/B/C/D acquisition (AoA)
# curve can be plotted on an identical x-axis. Full checkpoints (optimizer state
# included) are saved at each — the callback below converts these to global steps.
# Override for testing/tuning via env: MILESTONE_TOKENS_M="0.001,0.002" (millions).
_DEFAULT_MILESTONES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                       20, 30, 40, 50, 60, 70, 80, 90, 100,
                       150, 200, 250]
_env_milestones = os.environ.get("MILESTONE_TOKENS_M")
MILESTONE_TOKENS_M = ([float(x) for x in _env_milestones.split(",")]
                      if _env_milestones else _DEFAULT_MILESTONES)


class MilestoneSaver(TrainerCallback):
    """Force a full checkpoint save at a fixed set of global steps."""

    def __init__(self, target_steps: set[int]):
        self.targets = target_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.targets:
            control.should_save = True
        return control


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
    p.add_argument("--save_milestones", action="store_true",
                   help="save full trajectory checkpoints at the AoA token milestones "
                        "(for A/B/C/D trajectory runs); off = only the final model is saved "
                        "(for cheap error-bar seed runs)")
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

    callbacks = []
    if args.save_milestones:
        # Trajectory runs: one optimizer step consumes
        #   per_device_batch * seq_len * grad_accum * world_size  tokens.
        # Convert each token milestone to the global step it lands on.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        tokens_per_step = args.per_device_batch * args.seq_len * args.grad_accum * world_size
        target_steps = sorted({max(1, round(m * 1_000_000 / tokens_per_step))
                               for m in MILESTONE_TOKENS_M})
        print(f"milestone save steps ({tokens_per_step} tok/step): {target_steps}")
        callbacks.append(MilestoneSaver(set(target_steps)))
        # callback drives saving; keep all checkpoints (no total_limit) so the
        # full trajectory survives, and keep optimizer state (no save_only_model).
        save_strategy = "no"
    else:
        # Error-bar seed runs: only the final model is needed.
        save_strategy = "no"

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
        save_strategy=save_strategy,
        eval_strategy="epoch" if args.valid_file else "no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=lm["train"],
        eval_dataset=lm.get("validation"),
        data_collator=default_data_collator,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"done -> {args.output_dir}")


if __name__ == "__main__":
    main()
