# Simplification pipeline — plan, selector design, and cost

Forward plan after both feasibility gates passed (`simplification-pivot-findings.md`).
The experiment is three stages — **select → rewrite → assemble** — then train two
matched models and compare. This document scopes the selector and the cost; the
research question and arms are recorded at the end.

## The three stages

1. **Select.** Walk the corpus, score each sentence for complexity, and write a
   manifest of which sentences to rewrite (by source + line index, so the build
   is deterministic and re-runnable).
2. **Rewrite.** Send the selected sentences to an LLM with the simplification
   prompt; cache every result keyed by the original text (so reruns are free and
   the rewrite set is auditable).
3. **Assemble.** Rebuild the corpus: selected lines replaced by their rewrites,
   everything else untouched. This mirrors the existing `make_swapped_corpus.py`
   (word-swap) but at sentence granularity. Verify the token count against the
   original so the arms stay matched.

## Selector design

The selector decides the slice. Two refinements over raw Flesch-Kincaid:

- **Discount proper nouns.** FK over-flags name-heavy bios (the artifact seen in
  the rewrite test). Before scoring, mask proper-noun spans so the grade reflects
  *syntactic* complexity, not "has a long foreign name." The etymology labeller
  already flags proper nouns (`likely_proper` in `etymology_labels_full.csv`),
  and spaCy NER is available as a second source.
- **Pair with the register classifier.** FK is the primary selection signal;
  the classifier's register score is logged alongside as the second axis agreed
  for the report (primary = classifier score, check = readability).

Eligibility (carried over from the exploration step): strip speaker tags
(`*CHI:`, `A:`), drop headers (`= ... =`) and very short utterances (< 6 words).

Output: `data/derived/rewrite_manifest.csv` — one row per selected sentence with
`source, line_idx, original, fk_grade, register_score`.

## Threshold choice

The threshold trades reach against intervention strength. Recommended default:
**FK ≥ 10** — broad reach (28.5% of tokens) while clearly targeting complex text.
FK ≥ 12 is the conservative option; FK ≥ 8 is aggressive.

| Threshold | Sentences | Words in slice | % of corpus |
|---|---|---|---|
| FK ≥ 8 | 74,481 | 3.13M | 41.2% |
| **FK ≥ 10** | **43,913** | **2.16M** | **28.5%** |
| FK ≥ 12 | 25,871 | 1.44M | 18.9% |

## Engine + cost (LOCKED 2026-06-20)

**Engine: Llama-3.3-70B-Instruct via OpenRouter.** Chosen in a head-to-head pilot
on the 50 complex sentences: it cut FK grade harder than the hand baseline
(14.5→3.8) while *preserving meaning by splitting* dense lines into short ones.
Qwen2.5-7B was rejected — it *compresses* and drops facts on hard inputs (e.g.
collapsed a UN-resolution sentence and lost the resolution). The OpenRouter key
lives in the macOS keychain (`security find-generic-password -s
"llmpractical-course-api-key" -w`), pulled at runtime, never written to disk.

**Selected slice (after proper-noun discount + bracket filter): 31,860 lines /
1.74M words.** The proper-noun mask dropped ~10,900 name-heavy false positives;
a bracket filter dropped ~970 pure-annotation lines (CHILDES transcriber
comments, stage directions) that are metadata, not language.

Cost per 1M tokens on OpenRouter Llama-3.3-70B ≈ $0.10 in / $0.30 out. The full
FK≥10 slice (~4.8M input / ~2.6M output tokens) costs **~$1.50** — a fraction of
the €10 budget, leaving room for re-runs and the FK≥8 "rewrite everything" arm.
The real cost of the experiment is GPU time for training, which the course
provides.

## The experiment (recorded, not yet locked)

- **Research question:** does simplifying the high-register slice of a
  developmentally-plausible corpus improve sample-efficient pretraining?
- **Arms (phase 1):** A = original 10M corpus (control); B = surgical rewrite
  (FK-selected complex sentences simplified), token-matched to A. Add arm
  C = "rewrite everything" later only if time/compute allow — it isolates
  whether *surgical* beats blanket simplification.
- **"Simpler" measure:** primary = register-classifier score; check = FK grade.
- **Evaluation:** start with the cheapest harness contrast (BLiMP), A vs B.
- **Honesty note:** this reverses the deliberate "no whole-sentence rewriting"
  choice in `register-findings.md`. That choice was right for the old RQ (isolate
  register *net of* other changes); the new RQ *wants* length/grammar to change,
  because that change is the simplification. Disclosed in the report.

## Open risks

- Reach ≠ effect: a null result is possible and is itself a valid finding.
- Rewrite drift at scale: spot-audit a random sample of the full rewrite set for
  meaning preservation, not just the 50-sentence pilot.
- Distribution shift: rewriting only the complex slice could make the corpus
  stylistically lumpy (simple-native vs simplified-by-model). The "rewrite
  everything" arm (C) is the control for that, if we get to it.
