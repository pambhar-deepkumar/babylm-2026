"""Selector — find the high-register lines to rewrite, write the manifest.

The unit is a corpus *line* (one item per line in the source files; for the
bookish sources a line can be a short paragraph — we simplify the whole line).

Selection = Flesch-Kincaid grade >= FK_MIN, but on a *proper-noun-discounted*
score so name-heavy bios aren't over-flagged (the pilot artifact). Masking only
ever lowers FK, so a line below FK_MIN raw can never cross it after masking —
we therefore run the (slower) spaCy NER pass only on raw-FK>=FK_MIN candidates.

Output: data/derived/rewrite_manifest.csv with one row per selected line
(source, line_idx, text, fk_raw, fk_masked, n_words). line_idx is the 0-based
line number in the source file, so the assembler can map rewrites back exactly.

Usage: PYTHONPATH=src python scripts/build_manifest.py
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import pandas as pd
import spacy
import textstat

SNAP = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--BabyLM-community--BabyLM-2026-Strict-Small/snapshots/*/"
    )
)[0]
SOURCES = ["childes", "simple_wiki", "gutenberg", "open_subtitles", "bnc_spoken", "switchboard"]
FK_MIN = 10
MIN_WORDS = 6
OUT = Path("data/derived/rewrite_manifest.csv")

TAG = re.compile(r"^(\*?[A-Za-z]{1,4}\d?:)\t?")
MASK_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
}


def clean(line: str) -> str:
    return TAG.sub("", line.strip()).strip()


def mask(text: str, doc) -> str:
    """Replace proper-noun entity spans with a short neutral token ('Sam')."""
    out, last = [], 0
    for ent in doc.ents:
        if ent.label_ in MASK_LABELS:
            out.append(text[last:ent.start_char])
            out.append("Sam")
            last = ent.end_char
    out.append(text[last:])
    return "".join(out)


def main() -> None:
    # Pass 1: cheap raw-FK scan over every eligible line; keep candidates.
    candidates = []          # (source, line_idx, text, fk_raw, n_words)
    total = 0
    for src in SOURCES:
        with open(f"{SNAP}{src}.train.txt", encoding="utf-8") as fh:
            for idx, raw in enumerate(fh):
                s = clean(raw)
                if not s or s.startswith("=") or len(s.split()) < MIN_WORDS:
                    continue
                # Drop pure annotation lines (CHILDES transcriber comments,
                # stage directions): a whole line wrapped in [ ] is metadata,
                # not language a learner hears — don't rewrite it.
                if s.startswith("[") and s.endswith("]"):
                    continue
                total += 1
                fk = textstat.flesch_kincaid_grade(s)
                if fk >= FK_MIN:
                    candidates.append((src, idx, s, round(fk, 1), len(s.split())))
        print(f"  {src:14s} scanned")
    print(f"eligible lines: {total:,}; raw-FK>={FK_MIN} candidates: {len(candidates):,}")

    # Pass 2: proper-noun-masked FK on candidates only (masking can only lower FK).
    print("running NER mask on candidates...")
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    texts = [c[2] for c in candidates]
    rows = []
    for (src, idx, text, fk_raw, nw), doc in zip(candidates, nlp.pipe(texts, batch_size=256)):
        fk_masked = round(textstat.flesch_kincaid_grade(mask(text, doc)), 1)
        if fk_masked >= FK_MIN:
            rows.append(
                {"source": src, "line_idx": idx, "text": text,
                 "fk_raw": fk_raw, "fk_masked": fk_masked, "n_words": nw}
            )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    dropped = len(candidates) - len(df)
    print(f"\nselected (masked FK>={FK_MIN}): {len(df):,} lines, {df['n_words'].sum():,} words")
    print(f"proper-noun discount dropped {dropped:,} name-heavy candidates")
    print("by source:")
    print(df.groupby("source").agg(lines=("text", "size"), words=("n_words", "sum")).to_string())
    print(f"\nmanifest -> {OUT}")


if __name__ == "__main__":
    main()
