# From word-swap to sentence-simplification: does the new angle have legs?

A plain-language record of the two feasibility checks that justified pivoting the
project from *word-origin swapping* to *sentence-level simplification*. Both
checks passed. Intended as raw material for the report and for briefing the TA.

## Why we pivoted

The original idea — does plain (Germanic) vs fancy (Latinate) vocabulary change
what a small model learns — ran into a wall (see `register-findings.md`):

1. Word origin is almost entirely **confounded with frequency** in this corpus.
2. Once frequency is controlled, the word-origin effect is **essentially zero**.
3. A clean word-for-word swap could touch only **~0.27% of the corpus** — far
   too little to move any benchmark.

The wall was really about *reach*: the lever was too small to pull. So we changed
the lever. Instead of swapping single words, **simplify whole sentences** (with
an LLM where needed): shorten them, break up complex grammar, prefer common
words. This is closer to how a caregiver naturally pitches language to a child —
the developmentally-plausible framing the BabyLM challenge is built around.

Two questions had to be answered before committing: (1) is there enough complex
text to be worth simplifying, and (2) can an LLM actually simplify it well?

## Gate 1 — is there enough complex text? (reach)

We scored every eligible sentence in the corpus (435,986 sentences, 7.58M words,
after dropping very short utterances) for reading difficulty using the
Flesch-Kincaid grade level. Unlike the word-origin lever, complexity is
plentiful:

| Target (FK grade ≥) | Sentences | Words | Share of corpus |
|---|---|---|---|
| 8 (moderately complex) | 74,481 | 3.13M | **41.2%** |
| 10 (complex) | 43,913 | 2.16M | **28.5%** |
| 12 (genuinely complex) | 25,871 | 1.44M | **18.9%** |

That is **~70–150× the reach** of the 0.27% word-swap ceiling. The complexity is
concentrated by source, which makes a clean story: the speech-like sources
(`childes`, `open_subtitles`, `switchboard`) are already simple (FK ≈ 2–4) —
that is the child-directed speech our hypothesis is about — while the bookish
sources (`simple_wiki`, `gutenberg`, `bnc_spoken`) carry the complexity. So
"simplify the high-register slice" really means "make the encyclopedic/literary
text read more like everyday speech."

## Gate 2 — can an LLM simplify it well? (quality)

We sampled 50 sentences from the genuinely-complex tail (FK ≥ 12), including
dense Wikipedia biographies, literary prose, and garbled spoken-transcript
fragments, and rewrote each as simpler, child-directed English with the
instruction to keep all facts and add nothing.

| Measure | Result |
|---|---|
| Mean FK grade | **14.5 → 6.1** (drop of 8.4 grade levels) |
| Sentences made simpler | **100%** |
| Mean length | 27 → 27 words (unchanged) |
| Meaning preserved | Yes (manual check; facts/names/dates intact) |

Two things worth noting:

- **Length stayed flat.** Difficulty dropped not by deleting content but by
  *splitting* one dense sentence into several short ones. This is good news for
  the experiment: simplification will not shrink the token count much, so the
  training arms can be matched on tokens cleanly.
- **The few "still complex" rewrites are artifacts.** A couple stayed above
  FK 12 only because of long foreign proper nouns (e.g. "Ricardo Miró National
  Literature Contest") that inflate the grade score without real syntactic
  complexity. The selector should discount proper nouns (see the plan).

Full before/after data: `data/derived/rewrite_test.csv`.

## Caveats (stated plainly)

- **Reach existing is not the same as the intervention helping.** These gates
  only show the lever is big enough and the rewrites are good. Whether simpler
  input actually improves sample-efficient learning is the experiment itself.
- **FK grade is a proxy.** It is length- and syllable-based and over-flags
  name-heavy text. We pair it with the register classifier as a second axis.
- **The rewriter injects outside knowledge.** Using a large LLM to rewrite
  training data is a form of distillation. This is acceptable here because the
  project is an analysis, not a leaderboard entry; it will be disclosed in the
  report.

## Bottom line

The pivot clears both gates the old idea failed. There is a large, well-defined
slice of complex text (~19–41% of the corpus depending on threshold), and an LLM
simplifies it sharply (≈8 grade levels) while preserving meaning. The project is
worth taking to a real train-and-compare experiment. The forward plan and its
cost are in `simplification-pipeline-plan.md`.
