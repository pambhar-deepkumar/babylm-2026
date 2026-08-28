"""Quantify the magnitude of the simplification: readability + etymological
register on the original vs the simplified corpus.

Reports two views:
  - Intervention: the changed lines only (original -> rewrite), the effect where applied.
  - Whole corpus: original vs simplified, the net shift to the training data.

Metrics: Flesch-Kincaid grade, Flesch reading ease, Latinate share (% of
content words labelled Latinate vs Germanic, via the etymology labeller),
mean words/line.

Usage:
  PYTHONPATH=src python scripts/register/measure_simplification.py                   # Arm B
  PYTHONPATH=src python scripts/register/measure_simplification.py --variant register  # Arm C
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import pandas as pd
import textstat

random.seed(0)

SNAP = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--BabyLM-community--BabyLM-2026-Strict-Small/snapshots/*/"
    )
)[0]
ORDER = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
# Per-variant rewrite source + assembled corpus. simplify=Arm B, register=Arm C.
REWRITES_MAP = {
    "simplify": "data/derived/rewrites.jsonl",
    "register": "data/derived/rewrites_register.jsonl",
}
SIMPL_MAP = {
    "simplify": "data/bb26_simplified.train",
    "register": "data/bb26_register.train",
}
SIMPL = SIMPL_MAP["simplify"]       # set per --variant in main()
REWRITES = REWRITES_MAP["simplify"]
LABELS = "data/derived/etymology_labels_full.csv"
FK_SAMPLE = 40000   # lines sampled for the whole-corpus readability means

TAG = re.compile(r"^(\*?[A-Za-z]{1,4}\d?:)\t?")
WORD = re.compile(r"[a-z]+")


def clean(s: str) -> str:
    return TAG.sub("", s.strip()).strip()


def load_labels() -> dict[str, str]:
    df = pd.read_csv(LABELS, usecols=["word", "etym_label"])
    return dict(zip(df["word"].astype(str), df["etym_label"].astype(str)))


def latinate_share(texts, lab: dict[str, str]) -> tuple[float, int, int]:
    nl = ng = 0
    for t in texts:
        for w in WORD.findall(t.lower()):
            l = lab.get(w)
            if l == "Latinate":
                nl += 1
            elif l == "Germanic":
                ng += 1
    share = nl / (nl + ng) if (nl + ng) else float("nan")
    return share, nl, ng


def fk_mean(texts) -> tuple[float, float, float]:
    elig = [t for t in texts if len(t.split()) >= 6]
    if not elig:
        return float("nan"), float("nan"), float("nan")
    fk = [textstat.flesch_kincaid_grade(t) for t in elig]
    ease = [textstat.flesch_reading_ease(t) for t in elig]
    wpl = [len(t.split()) for t in elig]
    return sum(fk) / len(fk), sum(ease) / len(ease), sum(wpl) / len(wpl)


def row(name, texts, lab):
    fk, ease, wpl = fk_mean(texts)
    share, _, _ = latinate_share(texts, lab)
    return {"set": name, "fk_grade": round(fk, 1), "flesch_ease": round(ease, 1),
            "latinate_%": round(share * 100, 1), "words/line": round(wpl, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=REWRITES_MAP, default="simplify",
                    help="which rewrites/corpus to measure (simplify=Arm B, register=Arm C)")
    args = ap.parse_args()
    global REWRITES, SIMPL
    REWRITES = REWRITES_MAP[args.variant]
    SIMPL = SIMPL_MAP[args.variant]
    print(f"variant: {args.variant}  (rewrites={REWRITES})\n")

    lab = load_labels()

    # --- Intervention: the changed lines only (paired original -> rewrite) ---
    rw = [json.loads(l) for l in open(REWRITES, encoding="utf-8")]
    orig_slice = [r["original"] for r in rw]
    new_slice = [r["rewrite"] for r in rw]
    print(f"intervention: {len(rw):,} changed lines\n")
    inter = pd.DataFrame([row("original (changed lines)", orig_slice, lab),
                          row("simplified (rewrites)", new_slice, lab)])
    print("=== INTERVENTION (the changed lines only) ===")
    print(inter.to_string(index=False))

    # --- Whole corpus: original (reconstructed) vs simplified ---
    if not os.path.exists(SIMPL):
        print(f"\n(whole-corpus view skipped: {SIMPL} not assembled yet — "
              f"run make_simplified_corpus.py --variant {args.variant})")
        return
    original = []
    for src in ORDER:
        with open(f"{SNAP}{src}.train.txt", encoding="utf-8") as fh:
            original.extend(clean(x) for x in fh)
    simplified = [clean(x) for x in open(SIMPL, encoding="utf-8")]
    assert len(original) == len(simplified), (len(original), len(simplified))
    n = len(original)

    # etymology over the FULL corpus (cheap); readability over a shared sample
    idx = random.sample(range(n), min(FK_SAMPLE, n))
    o_samp = [original[i] for i in idx]
    s_samp = [simplified[i] for i in idx]
    o_fk = fk_mean(o_samp)
    s_fk = fk_mean(s_samp)
    o_share = latinate_share(original, lab)[0]
    s_share = latinate_share(simplified, lab)[0]

    whole = pd.DataFrame([
        {"set": "original (full corpus)", "fk_grade": round(o_fk[0], 1),
         "flesch_ease": round(o_fk[1], 1), "latinate_%": round(o_share * 100, 2),
         "words/line": round(o_fk[2], 1)},
        {"set": "simplified (full corpus)", "fk_grade": round(s_fk[0], 1),
         "flesch_ease": round(s_fk[1], 1), "latinate_%": round(s_share * 100, 2),
         "words/line": round(s_fk[2], 1)},
    ])
    print(f"\n=== WHOLE CORPUS ({n:,} lines; readability on {len(idx):,}-line shared sample) ===")
    print(whole.to_string(index=False))
    print("\n(higher Flesch ease = easier; lower FK grade = easier; lower Latinate% = plainer/more Germanic)")


if __name__ == "__main__":
    main()
