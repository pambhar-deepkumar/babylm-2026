"""Rewriter — simplify the manifest's lines via Llama-3.3-70B on OpenRouter.

Concurrent (thread pool) and resumable: every completed rewrite is appended to a
JSONL checkpoint keyed by "source:line_idx", so a re-run skips finished lines and
only does what's left. The treatment is defined by SYSTEM (the fixed instruction,
sent as the system prompt) + the line as the user message — quote SYSTEM verbatim
in the report; it IS the experiment's intervention.

Two treatments via --variant (each writes its own checkpoint):
  simplify (Arm B, default) -> data/derived/rewrites.jsonl
  register (Arm C)          -> data/derived/rewrites_register.jsonl

Key from the keychain. Run:
  OPENROUTER_API_KEY=$(security find-generic-password -s "llmpractical-course-api-key" -w) \
  PYTHONPATH=src python scripts/rewrite_corpus.py                       # Arm B, full run
  ... scripts/rewrite_corpus.py --variant register                     # Arm C, full run
  ... scripts/rewrite_corpus.py --variant register --limit 200         # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

MODEL = "meta-llama/llama-3.3-70b-instruct"
MANIFEST = Path("data/derived/rewrite_manifest.csv")
WORKERS = 12
MAX_RETRIES = 4

# Each variant is a distinct treatment (its SYSTEM prompt IS the intervention) and
# writes to its own checkpoint so the corpora never collide. Quote the chosen
# prompt verbatim in the report.
#   simplify (Arm B): lower register AND syntax — the bundled simplification.
#   register (Arm C): swap formal/Latinate words for plain ones, syntax untouched —
#                     isolates the lexical-register lever for a clean B-vs-C contrast.
PROMPTS = {
    "simplify": (
        "You rewrite text into simple, developmentally-plausible English, the way a "
        "caregiver speaks to a young child. Use short sentences, common everyday words, "
        "and simple grammar; break one long sentence into several short ones. Preserve "
        "the original meaning and every fact. Do not add information, opinions, or "
        "commentary, and do not omit facts. Reply with only the rewritten text."
    ),
    "register": (
        "You rewrite text by replacing formal, literary, or Latinate words with the "
        "plain, everyday words a young child would hear (for example: purchase -> buy, "
        "comprehend -> understand, assist -> help). Keep the sentence's meaning, "
        "structure, and length exactly the same: do not split, shorten, or reorder "
        "sentences, and do not change the grammar. Do not replace words that carry "
        "specific, technical, or factual meaning (names, places, numbers, and "
        "specialist terms such as 'toxins' or 'regional'); swap a word only when the "
        "plain everyday synonym means exactly the same thing. Do not add or remove any "
        "facts. Reply with only the rewritten text."
    ),
    # register_strong (Arm D): the aggressive end of the register spectrum. Same axis as
    # Arm C (lexical register, syntax untouched) but dialled up — swap EVERY Latinate word,
    # accept rough synonyms, prefer phrasal verbs. Trades a little precision for a bigger
    # register shift. Still keeps structure, length, and facts (names/places/numbers) fixed,
    # so A -> C -> D stays a clean vocabulary-only dose-response.
    "register_strong": (
        "You rewrite text into the plainest possible English, using only the simple, "
        "everyday words a young child would know. Replace every formal, literary, or "
        "Latinate word with its most common plain equivalent, and prefer short "
        "Anglo-Saxon words and phrasal verbs (for example: require -> need, obtain -> "
        "get, demonstrate -> show, sufficient -> enough, numerous -> many, tolerate -> "
        "put up with, assist -> help, comprehend -> understand, purchase -> buy, "
        "approximately -> about, additional -> more). Swap a word even when the plain "
        "version is only roughly equivalent, as long as the sentence still means the "
        "same thing overall. Keep the sentence's structure and length the same: do not "
        "split, shorten, reorder, or change the grammar. Keep all names, places, and "
        "numbers exactly as written, and do not add or remove facts. Reply with only "
        "the rewritten text."
    ),
}
CKPTS = {
    "simplify": Path("data/derived/rewrites.jsonl"),
    "register": Path("data/derived/rewrites_register.jsonl"),
    "register_strong": Path("data/derived/rewrites_register_strong.jsonl"),
}

# Set per --variant in main(); module-level defaults keep the helpers importable.
SYSTEM = PROMPTS["simplify"]
CKPT = CKPTS["simplify"]

_lock = threading.Lock()


def client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set (pull from keychain — see docstring).")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def load_done() -> set[str]:
    if not CKPT.exists():
        return set()
    done = set()
    with open(CKPT, encoding="utf-8") as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def rewrite(cli: OpenAI, text: str) -> str:
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = cli.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
            )
            out = (resp.choices[0].message.content or "").strip()
            if out:
                return out
            last = "empty response"
        except Exception as e:  # noqa: BLE001
            last = str(e)
    raise RuntimeError(last)


def append(record: dict) -> None:
    with _lock, open(CKPT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=PROMPTS, default="simplify",
                    help="which rewrite treatment to apply (simplify=Arm B, register=Arm C)")
    ap.add_argument("--limit", type=int, default=0, help="only process first N pending (smoke test)")
    args = ap.parse_args()

    global SYSTEM, CKPT
    SYSTEM = PROMPTS[args.variant]
    CKPT = CKPTS[args.variant]
    print(f"variant: {args.variant}  ->  {CKPT}")

    man = pd.read_csv(MANIFEST)
    man["key"] = man["source"] + ":" + man["line_idx"].astype(str)
    done = load_done()
    pending = man[~man["key"].isin(done)]
    if args.limit:
        # Sample randomly so a smoke test is representative across sources,
        # not just the first source in manifest order.
        pending = pending.sample(min(args.limit, len(pending)), random_state=0)
    print(f"manifest {len(man):,} | done {len(done):,} | this run {len(pending):,} (model {MODEL})")
    if pending.empty:
        return

    cli = client()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {
            ex.submit(rewrite, cli, row.text): row
            for row in pending.itertuples()
        }
        for i, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            try:
                rw = fut.result()
                append({"key": row.key, "source": row.source, "line_idx": int(row.line_idx),
                        "original": row.text, "rewrite": rw})
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"\n  FAILED {row.key}: {e}")
            if i % 25 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)}  ok={n_ok} err={n_err}", end="\r")
    print(f"\ndone: {n_ok} ok, {n_err} errored -> {CKPT}")
    if n_err:
        print("re-run the same command to retry the errored lines (resumable).")


if __name__ == "__main__":
    main()
