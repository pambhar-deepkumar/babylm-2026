"""Assembler — build the Arm B (simplified) corpus by swapping rewritten lines in.

Reconstructs the corpus exactly as the team's bb26_en.train was built (raw
concatenation of the six source files, in this order, with speaker tags and
bracket lines preserved), then replaces each manifest line with its rewrite.

Safety:
  - Per-line check: before swapping line N of a source, verify clean(raw line N)
    == the original we stored in rewrites.jsonl. Any mismatch aborts (a wrong
    line index can't silently corrupt the corpus).
  - Base checksum: also hashes the no-swap reconstruction and prints it, so we
    can confirm against `md5sum bb26_en.train` that our base is byte-identical
    to the original (unmodified) Arm A corpus.

Output (scp to the cluster's data/ dir):
  --variant simplify (default) -> data/bb26_simplified.train  (Arm B)
  --variant register           -> data/bb26_register.train    (Arm C)

Usage:
  PYTHONPATH=src python scripts/make_simplified_corpus.py
  PYTHONPATH=src python scripts/make_simplified_corpus.py --variant register
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from pathlib import Path

SNAP = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--BabyLM-community--BabyLM-2026-Strict-Small/snapshots/*/"
    )
)[0]
# Concatenation order confirmed against the server's bb26_en.train (line 1 =
# bnc_spoken, line 65221 = childes start; total 1,104,106 lines).
ORDER = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
# Per-variant rewrite source -> output corpus. simplify=Arm B, register=Arm C.
REWRITES_MAP = {
    "simplify": Path("data/derived/rewrites.jsonl"),
    "register": Path("data/derived/rewrites_register.jsonl"),
    "register_strong": Path("data/derived/rewrites_register_strong.jsonl"),
}
OUT_MAP = {
    "simplify": Path("data/bb26_simplified.train"),
    "register": Path("data/bb26_register.train"),
    "register_strong": Path("data/bb26_register_strong.train"),
}
# Set per --variant in main(); defaults preserve the original behaviour.
REWRITES = REWRITES_MAP["simplify"]
OUT = OUT_MAP["simplify"]

TAG = re.compile(r"^(\*?[A-Za-z]{1,4}\d?:)\t?")
WS = re.compile(r"\s+")


def clean(line: str) -> str:
    return TAG.sub("", line.strip()).strip()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=REWRITES_MAP, default="simplify",
                    help="which rewrites to assemble (simplify=Arm B, register=Arm C)")
    args = ap.parse_args()
    global REWRITES, OUT
    REWRITES = REWRITES_MAP[args.variant]
    OUT = OUT_MAP[args.variant]
    print(f"variant: {args.variant}  ->  {OUT}")

    # Build swap map keyed by (source, line_idx) -> (stored_original, rewrite).
    swaps: dict[tuple[str, int], tuple[str, str]] = {}
    with open(REWRITES, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            # collapse any internal newlines so each swapped entry stays ONE line
            rw = WS.sub(" ", r["rewrite"].replace("\n", " ")).strip()
            swaps[(r["source"], int(r["line_idx"]))] = (r["original"], rw)
    print(f"rewrites to apply: {len(swaps):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    base_hash = hashlib.md5()      # hash of the no-swap reconstruction (== original corpus)
    total = swapped = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for src in ORDER:
            path = f"{SNAP}{src}.train.txt"
            with open(path, encoding="utf-8") as fh:
                for idx, raw in enumerate(fh):
                    base_hash.update(raw.encode("utf-8"))
                    total += 1
                    key = (src, idx)
                    if key in swaps:
                        stored, rw = swaps[key]
                        if clean(raw) != stored:
                            raise SystemExit(
                                f"MISMATCH at {src}:{idx}\n  file:   {clean(raw)!r}\n"
                                f"  stored: {stored!r}\naborting — line mapping is wrong."
                            )
                        out.write(rw + "\n")
                        swapped += 1
                    else:
                        out.write(raw if raw.endswith("\n") else raw + "\n")

    print(f"total lines:   {total:,}")
    print(f"lines swapped: {swapped:,}  (expected {len(swaps):,})")
    print(f"base md5 (no-swap reconstruction): {base_hash.hexdigest()}")
    print("  -> compare to `md5sum` of your unmodified BabyLM-2026 corpus")
    print(f"output -> {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")
    if swapped != len(swaps):
        raise SystemExit("ERROR: not every rewrite was applied — check line indices.")


if __name__ == "__main__":
    main()
