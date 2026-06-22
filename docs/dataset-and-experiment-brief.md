# Simplified Dataset & Experiment Brief

## Simplified dataset (what it is)
We made a "simpler" version of the official BabyLM 2026 Strict-Small corpus
(10M words of English). Most of the text is unchanged. We only rewrote the
hardest, most complex sentences into plain, child-directed English. The idea
comes from how children learn: they hear simple speech, so a small model might
also learn better from simpler text.

## Why we pivoted here (background)
This project started as a different question: does the *origin* of words matter?
English has a plain everyday layer with Germanic roots (understand, buy, freedom)
and a fancier layer borrowed from Latin and French (comprehend, purchase,
liberty). We wanted to test whether a small model learns more efficiently from
plainer, more Germanic text.

To study that we first had to label every word as plain or fancy. Our first
labeller could confidently tag only a tiny slice of the text, so we rebuilt it
around a large etymology database, which covered far more words at high accuracy.
But two problems appeared. First, "fancy" words are almost always *rare* words,
so word origin is tangled up with word frequency (you can't tell them apart).
Second, to test it by experiment we would swap fancy words for plain synonyms
throughout the corpus, using a published list of plain-English pairs, but a clean
one-for-one swap could only change about 0.27% of the text. Most fancy words have
no clean single-word plain equivalent, and the ones that do are themselves rare.
That is far too little to change anything measurable.

So we pivoted. Instead of swapping individual words (a tiny lever), we simplify
whole complex sentences into plain, child-directed English (a much bigger lever,
~19% of the text). Helpfully, doing this still shifts the vocabulary toward the
plain Germanic core, so we keep the original word-origin angle as a side
measurement, but through a change big enough to matter.

| step | what we measured | number |
|---|---|---|
| First origin-labeller | share of corpus words confidently labelled | 0.42% (707 Latinate types) |
| Rebuilt labeller (etymology DB) | share of corpus words labelled | ~9.5% (16,386 Latinate types) |
| Labeller accuracy | precision vs a reference gold list | 98.9% |
| Swap list | plain-English Latinate→Germanic pairs found in the corpus | 278 pairs |
| Frequency tangle | pairs where the Latinate word is the rarer one | 98% (median 38 vs 1,967 uses, ~50x) |
| Direct word-swap reach | share of corpus a clean one-word swap could change | ~0.27% (≈1 word in 370) |
| Sentence-simplification reach (our pivot) | share of corpus we actually rewrote | ~19% |

## How we made it
1. Score every line for reading difficulty (Flesch-Kincaid grade), discounting
   long proper names so biographies don't look harder than they are.
2. Select the hard lines (grade ≥ 10): about 32,000 lines, ~1.7M words
   (~19% of the corpus).
3. Rewrite each hard line with an LLM (Llama-3.3-70B) into short, simple
   sentences, keeping all the facts.
4. Put the rewrites back in place; every other line is left exactly as-is.

The changed lines dropped from reading grade ~15 to ~3, with meaning preserved.
The result is a corpus identical to the original except those ~32k lines.

## Simplification magnitude (manipulation check)
We measured how much we actually simplified, on two independent axes: reading
difficulty and word origin (Latinate vs plain Germanic vocabulary).

On the changed lines (where the rewrite was applied):

| metric | original | simplified |
|---|---|---|
| reading grade (Flesch-Kincaid) | 14.8 | 2.6 |
| reading ease (Flesch) | 42.8 | 88.8 |
| Latinate words (% of content words) | 48.3 | 32.1 |
| words per line | 54.5 | 59.2 |

Across the whole corpus (only ~19% of lines changed, so the shift is smaller):

| metric | original | simplified |
|---|---|---|
| reading grade | 4.6 | 3.7 |
| reading ease | 82.1 | 85.5 |
| Latinate words (%) | 36.4 | 33.3 |

How to read these:
- **Reading grade (Flesch-Kincaid):** the US school grade needed to read the
  text. Lower is easier. Roughly 1-5 = early primary, 6-8 = middle school,
  9-12 = high school, 13+ = college. (So 2.6 is a young child, 14.8 is college.)
- **Reading ease (Flesch):** a 0-100 score, higher is easier. Roughly 90-100
  very easy, 60-70 plain English, 30-50 difficult, below 30 very hard. (88.8 is
  "easy", 42.8 is "difficult".)
- **Latinate words (%):** share of meaningful words that come from Latin/French
  (formal, "fancy") rather than the plain Germanic core of English. Lower means
  plainer, more everyday vocabulary. There is no fixed target, it is relative:
  casual speech is low, academic writing is high.
- **Words per line:** average line length, not a difficulty score on its own. It
  rises slightly because rewrites split one long sentence into several short
  ones (more words, simpler structure), so the facts are kept.

Both axes agree: the rewrites are far easier to read and use plainer, more
Germanic vocabulary. This confirms the treatment did what we intended, before we
look at any model scores.

## Experiment design (A/B)
Same model, same training recipe, only the data differs:
- **Arm A** = original corpus.
- **Arm B** = simplified corpus.

Train a model on each and compare on the BabyLM 2026 tasks. Any score difference
is due to the simplification alone. Question: does simpler input help a small
model learn more from the same amount of data?

## The baseline
We use the official **2026 GPT-2 Strict-Small** baseline as Arm A (already
trained by the organizers on the original corpus). For Arm B we train an
identical GPT-2 (124M parameters, same tokenizer, AdamW, lr 5e-4, 20 epochs) on
our simplified corpus. Both are scored on the 2026 tasks (BLiMP, HellaSwag,
MultiBLiMP, WinoGrande, XStoryCloze); the headline metric is BLiMP. The official
baseline scores **BLiMP 0.6439** — that is the number to beat.
