"""Regenerate the etymology word lists with the combined labeller in
babylm_2026.register.etymology (LLM -> EtymDB -> lemma -> affix), proper nouns filtered.

Writes:
  data/derived/etymology_labels_full.csv      every corpus type, fully annotated
  data/derived/latinate_by_frequency.csv      all Latinate types, by frequency
  data/derived/latinate_common_by_frequency.csv  Latinate, proper nouns removed
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from babylm_2026.register.etymology import build_etymology_wordlist  # noqa: E402

CORPUS = ROOT / "data" / "babylm_2026" / "strict_small"
DERIVED = ROOT / "data" / "derived"

print("Labelling the full corpus vocabulary (EtymDB + lemma + affix + LLM)...")
df = build_etymology_wordlist(CORPUS)

total_tokens = int(round(df["freq"].sum()))
print(f"\n{len(df):,} types, {total_tokens:,} tokens")
print("\nLabel distribution:")
print(df["etym_label"].value_counts())
print("\nLabel source breakdown:")
print(df["etym_source"].value_counts())

full = DERIVED / "etymology_labels_full.csv"
df.to_csv(full, index=False)
print(f"\nwrote {full}")

lat = df[df["etym_label"] == "Latinate"].copy()
lat_tok = lat["freq"].sum()
print(f"\nLatinate: {len(lat):,} types, {lat_tok:,} tokens "
      f"({lat_tok / total_tokens:.2%} of corpus)")

lat[["word", "freq", "global_rank", "pct_of_corpus", "cap_ratio",
     "likely_proper", "etym_source"]].to_csv(
    DERIVED / "latinate_by_frequency.csv", index=False)

common = lat[~lat["likely_proper"]].copy()
common_tok = common["freq"].sum()
common[["word", "freq", "global_rank", "pct_of_corpus", "etym_source"]].to_csv(
    DERIVED / "latinate_common_by_frequency.csv", index=False)
print(f"proper nouns dropped: {len(lat) - len(common)}  "
      f"-> latinate_common: {len(common):,} types, {common_tok:,} tokens "
      f"({common_tok / total_tokens:.2%})")

print("\n--- top 40 common Latinate (proper nouns removed) ---")
for _, r in common.head(40).iterrows():
    print(f"  {r['word']:16}{r['freq']:>7}  ({r['etym_source']})")
