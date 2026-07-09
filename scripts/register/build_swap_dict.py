"""Build a clean fancy->plain swap dictionary and apply it to the corpus.

Curated, quality-first pairs (meaning-preserving, low-polysemy, same part of
speech), expanded to every inflected form with lemminflect so 'acquired' also
maps to 'got'. Replaces the noisy retext-simplify pairs.

Writes data/derived/swap_dict.csv and data/swapped/strict_small/*.train.txt,
and reports the intervention size + before/after examples.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import lemminflect

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "babylm_2026" / "strict_small"
OUT = ROOT / "data" / "swapped" / "strict_small"
OUT.mkdir(parents=True, exist_ok=True)

# (latinate, germanic) — curated, high-confidence, low polysemy.
VERBS = [
    ("comprehend", "understand"), ("acquire", "get"), ("obtain", "get"),
    ("require", "need"), ("desire", "want"), ("provide", "give"),
    ("receive", "get"), ("attempt", "try"), ("assist", "help"),
    ("construct", "build"), ("demonstrate", "show"), ("purchase", "buy"),
    ("depart", "leave"), ("permit", "let"), ("allow", "let"), ("request", "ask"),
    ("inquire", "ask"), ("reply", "answer"), ("respond", "answer"),
    ("conceal", "hide"), ("discover", "find"), ("select", "choose"),
    ("repair", "mend"), ("possess", "have"), ("conclude", "end"),
    ("commence", "start"), ("observe", "see"), ("remain", "stay"),
    ("retain", "keep"), ("maintain", "keep"), ("perform", "do"),
]
NOUNS = [
    ("infant", "baby"), ("beverage", "drink"), ("residence", "home"),
    ("error", "mistake"), ("illness", "sickness"), ("remainder", "rest"),
    ("assistance", "help"),
]
# adjectives / adverbs / modal — no inflection needed
FIXED = [
    ("enormous", "huge"), ("gigantic", "huge"), ("immense", "huge"),
    ("massive", "huge"), ("rapid", "fast"), ("difficult", "hard"),
    ("sufficient", "enough"), ("entire", "whole"), ("numerous", "many"),
    ("multiple", "many"), ("additional", "more"), ("ancient", "old"),
    ("previous", "earlier"), ("prior", "earlier"), ("initial", "first"),
    ("subsequent", "next"), ("fortunate", "lucky"), ("strange", "odd"),
    ("peculiar", "odd"), ("ill", "sick"), ("secure", "safe"), ("final", "last"),
    ("however", "but"), ("therefore", "so"), ("perhaps", "maybe"),
    ("frequently", "often"), ("previously", "before"),
    ("occasionally", "sometimes"), ("currently", "now"), ("shall", "will"),
]


def expand(lat, ger, upos, tags):
    li = lemminflect.getAllInflections(lat, upos=upos)
    gi = lemminflect.getAllInflections(ger, upos=upos)
    out = {}
    for t in tags:
        lf = li.get(t) or li.get("VBD")
        gf = gi.get(t) or gi.get("VBD")
        if lf and gf:
            out[lf[0].lower()] = gf[0].lower()
    return out


swap: dict[str, str] = {}
for lat, ger in VERBS:
    swap.update(expand(lat, ger, "VERB", ["VB", "VBP", "VBZ", "VBD", "VBG", "VBN"]))
for lat, ger in NOUNS:
    swap.update(expand(lat, ger, "NOUN", ["NN", "NNS"]))
for lat, ger in FIXED:
    swap[lat.lower()] = ger.lower()

import csv
with open(ROOT / "data" / "derived" / "swap_dict.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["latinate", "germanic"])
    w.writerows(sorted(swap.items()))
print(f"swap dictionary: {len(swap)} surface forms "
      f"(from {len(VERBS)+len(NOUNS)+len(FIXED)} curated lemma pairs)")

pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, swap), key=len, reverse=True)) + r")\b",
                     re.IGNORECASE)


def repl(m):
    w = m.group(0)
    g = swap[w.lower()]
    return g.capitalize() if w[:1].isupper() else g


TOKRE = re.compile(r"[a-z]+")
total = swapped = 0
per_word: dict[str, int] = {}
examples = []
for p in sorted(SRC.glob("*.train.txt")):
    text = p.read_text(encoding="utf-8")
    total += len(TOKRE.findall(text.lower()))

    def repl_count(m):
        per_word[m.group(0).lower()] = per_word.get(m.group(0).lower(), 0) + 1
        return repl(m)

    (OUT / p.name).write_text(pattern.sub(repl_count, text), encoding="utf-8")
    if len(examples) < 6:
        for line in text.splitlines():
            if pattern.search(line) and 30 < len(line) < 150:
                examples.append((line.strip(), pattern.sub(repl, line).strip()))
                if len(examples) >= 6:
                    break

swapped = sum(per_word.values())
print(f"tokens swapped: {swapped:,} of {total:,} = {swapped/total:.2%} of corpus")
print("\ntop swapped:")
for w, c in sorted(per_word.items(), key=lambda kv: -kv[1])[:18]:
    print(f"  {w:14} -> {swap[w]:10} {c:>5}x")
print("\n--- before / after ---")
for b, a in examples:
    print(f"  BEFORE: {b}\n  AFTER : {a}\n")
