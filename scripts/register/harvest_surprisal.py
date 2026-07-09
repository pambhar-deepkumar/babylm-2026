"""Harvest per-word surprisal across an arm's training checkpoints (AoA curve).

For each milestone checkpoint of one arm, load the model and compute the mean
surprisal (bits) of every target word over its fixed probe contexts, then write
a tidy CSV: one row per (checkpoint, word). Aggregation into Germanic-vs-Latinate
curves and the human-AoA correlation happens downstream from this CSV.

    python scripts/register/harvest_surprisal.py \
        --arm_dir output/traj_a --arm_name A \
        --probes data/aoa_probes.json --out results/register/aoa/traj_a_surprisal.csv

Surprisal of a target word = -log2 p(word tokens | left context), summed over the
word's sub-word tokens, averaged over its probe contexts. Lower = better acquired.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKENS_PER_STEP = 32768  # per_device_batch(32) * seq_len(1024) * grad_accum(1), single GPU


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arm_dir", required=True, help="e.g. output/traj_a")
    p.add_argument("--arm_name", required=True, help="short label, e.g. A / B / C / D")
    p.add_argument("--probes", default="data/aoa_probes.json")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="baseline_gpt2_2026")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def list_checkpoints(arm_dir: Path) -> list[tuple[int, Path]]:
    """(global_step, path) for every milestone checkpoint-* dir, in step order.

    The final root model is intentionally excluded: its true global_step is not
    recoverable from the saved dir, and the 22 milestone checkpoints already
    cover the trajectory to 250M tokens seen.
    """
    cks = []
    for d in arm_dir.glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", d.name)
        if m and (d / "model.safetensors").exists():
            cks.append((int(m.group(1)), d))
    cks.sort()
    return cks


@torch.no_grad()
def word_surprisals(model, tokenizer, probes, device, batch_size):
    """Return {word: (mean_bits, n)} over all probe contexts."""
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = 1  # <s> in the 2026 tokenizer
    ln2 = math.log(2.0)
    sums: dict[str, float] = {}
    ns: dict[str, int] = {}

    # precompute token spans once (tokenizer is shared across checkpoints)
    items = []
    for pr in probes:
        ctx_ids = tokenizer(pr["ctx_left"], add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(pr["ctx_left"] + pr["target"], add_special_tokens=False)["input_ids"]
        tgt_ids = full_ids[len(ctx_ids):]
        if not tgt_ids:  # target merged into context boundary; skip (rare)
            continue
        seq = [bos] + full_ids
        # positions predicting the target tokens: logits index = (len(seq)-len(tgt)-1 .. len(seq)-2)
        first = len(seq) - len(tgt_ids) - 1
        items.append((pr["word"], seq, first, tgt_ids))

    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        maxlen = max(len(s) for _, s, _, _ in chunk)
        inp = torch.full((len(chunk), maxlen), fill_value=tokenizer.pad_token_id or 3,
                         dtype=torch.long)
        att = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for r, (_, seq, _, _) in enumerate(chunk):
            inp[r, :len(seq)] = torch.tensor(seq)
            att[r, :len(seq)] = 1
        logits = model(input_ids=inp.to(device), attention_mask=att.to(device)).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        for r, (word, seq, first, tgt_ids) in enumerate(chunk):
            bits = 0.0
            for j, tid in enumerate(tgt_ids):
                bits += -logp[r, first + j, tid].item() / ln2
            sums[word] = sums.get(word, 0.0) + bits
            ns[word] = ns.get(word, 0) + 1
    return {w: (sums[w] / ns[w], ns[w]) for w in sums}


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    probes = json.load(open(args.probes, encoding="utf-8"))
    klass = {p["word"]: p["klass"] for p in probes}
    pair = {p["word"]: p["pair_id"] for p in probes}

    arm_dir = Path(args.arm_dir)
    cks = list_checkpoints(arm_dir)
    print(f"{args.arm_name}: {len(cks)} checkpoints, {len(probes)} probes, device={device}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "step", "tokens_seen", "word", "klass", "pair_id",
                    "surprisal_bits", "n"])
        for step, path in cks:
            model = AutoModelForCausalLM.from_pretrained(path).to(device).eval()
            res = word_surprisals(model, tokenizer, probes, device, args.batch_size)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            toks = step * TOKENS_PER_STEP
            for word, (bits, n) in sorted(res.items()):
                w.writerow([args.arm_name, step, toks, word, klass[word], pair[word],
                            f"{bits:.4f}", n])
            print(f"  ck {step}: {len(res)} words, "
                  f"mean surprisal {sum(b for b, _ in res.values())/len(res):.2f} bits")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
