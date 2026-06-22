"""Rewriter — simplify the manifest's lines via Llama-3.3-70B on OpenRouter.

Concurrent (thread pool) and resumable: every completed rewrite is appended to a
JSONL checkpoint keyed by "source:line_idx", so a re-run skips finished lines and
only does what's left. The treatment is defined by SYSTEM (the fixed instruction,
sent as the system prompt) + the line as the user message — quote SYSTEM verbatim
in the report; it IS the experiment's intervention.

Key from the keychain. Run:
  OPENROUTER_API_KEY=$(security find-generic-password -s "llmpractical-course-api-key" -w) \
  PYTHONPATH=src python scripts/rewrite_corpus.py            # full run
  ... scripts/rewrite_corpus.py --limit 200                  # smoke test

Outputs data/derived/rewrites.jsonl (checkpoint, one record per line).
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
CKPT = Path("data/derived/rewrites.jsonl")
WORKERS = 12
MAX_RETRIES = 4

SYSTEM = (
    "You rewrite text into simple, developmentally-plausible English, the way a "
    "caregiver speaks to a young child. Use short sentences, common everyday words, "
    "and simple grammar; break one long sentence into several short ones. Preserve "
    "the original meaning and every fact. Do not add information, opinions, or "
    "commentary, and do not omit facts. Reply with only the rewritten text."
)

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
    ap.add_argument("--limit", type=int, default=0, help="only process first N pending (smoke test)")
    args = ap.parse_args()

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
