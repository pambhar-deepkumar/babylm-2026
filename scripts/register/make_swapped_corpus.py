"""Build the 'germanicised' corpus: swap fancy (Latinate) words for plain
(Germanic) synonyms, using the existing pair list. Deliberately simple:
exact whole-word replacement, no frequency matching, no inflection expansion.

Writes:
  data/derived/swap_dict.csv            the cleaned pairs actually used
  data/swapped/strict_small/*.train.txt the swapped corpus (mirrors the original)
and prints the intervention size + before/after examples.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SRC = ROOT / "data" / "babylm_2026" / "strict_small"
OUT = ROOT / "data" / "swapped" / "strict_small"
OUT.mkdir(parents=True, exist_ok=True)

# Targets that would break grammar / change part of speech if substituted.
_BAD_TARGETS = {"of", "is", "on", "with", "about", "there", "here", "their",
                "at", "in", "to", "by", "a", "an", "the", "as"}

pairs = pd.read_csv(ROOT / "data" / "derived" / "candidate_pairs.csv")
pairs = pairs[(~pairs["labeler_conflict"]) & (pairs["both_in_corpus"])]
pairs = pairs[~pairs["germanic"].isin(_BAD_TARGETS)]
pairs = pairs[pairs["lat_len"] >= 4]
# one target per latinate source (keep the most frequent germanic option)
pairs = pairs.sort_values("ger_freq", ascending=False).drop_duplicates("latinate")
swap = dict(zip(pairs["latinate"].str.lower(), pairs["germanic"].str.lower()))

pd.DataFrame(sorted(swap.items()), columns=["latinate", "germanic"]).to_csv(
    ROOT / "data" / "derived" / "swap_dict.csv", index=False)
print(f"swap dictionary: {len(swap)} pairs")

# one combined word-boundary regex; case-preserving replacement
pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, swap), key=len, reverse=True)) + r")\b",
                     re.IGNORECASE)


def repl(m):
    w = m.group(0)
    g = swap[w.lower()]
    return g.capitalize() if w[:1].isupper() else g


total_tokens = swapped = 0
per_word = {}
examples = []
TOKRE = re.compile(r"[a-z]+")

for p in sorted(SRC.glob("*.train.txt")):
    text = p.read_text(encoding="utf-8")
    total_tokens += len(TOKRE.findall(text.lower()))

    def repl_count(m):
        per_word[m.group(0).lower()] = per_word.get(m.group(0).lower(), 0) + 1
        return repl(m)

    new = pattern.sub(repl_count, text)
    (OUT / p.name).write_text(new, encoding="utf-8")

    # grab a couple of before/after example lines from this source
    if len(examples) < 6:
        for line in text.splitlines():
            if pattern.search(line) and 30 < len(line) < 160:
                examples.append((line.strip(), pattern.sub(repl, line).strip()))
                if len(examples) >= 6:
                    break

swapped = sum(per_word.values())
print(f"tokens swapped: {swapped:,} of {total_tokens:,}  = {swapped/total_tokens:.2%} of corpus")
print(f"\ntop swapped words:")
for w, c in sorted(per_word.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {w:14} -> {swap[w]:10} {c:>5}x")

print("\n--- before / after examples ---")
for before, after in examples:
    print(f"  BEFORE: {before}")
    print(f"  AFTER : {after}\n")

print(f"swapped corpus written to {OUT}")
