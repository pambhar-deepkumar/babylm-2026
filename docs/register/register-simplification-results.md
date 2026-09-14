# Register & Simplification — Runs and Results

**BabyLM 2026, Strict-Small (English, 10M words).** Controlled study: does lowering the
*register* of the hardest text in the pretraining corpus improve sample-efficient learning?
We hold everything fixed except the corpus and compare the official baseline against three
treatments along a register/simplification spectrum.

**Headline:** No. Across a graded spectrum (mild → aggressive register), benchmark scores do
not improve and show no dose-response. The one apparent single-seed win disappeared under
multi-seed replication. Clean null result.

---

## Setup (identical across all arms)

- **Model:** GPT-2 (decoder-only), 12 layers / 12 heads / hidden 768, context 1024,
  tokenizer vocab 16,384 (≈98M parameters). Architecture + tokenizer loaded from the official
  2026 baseline; weights trained from scratch.
- **Recipe:** AdamW, lr 5e-4, cosine schedule, 20 epochs, batch 32 sequences × 1024 tokens,
  bf16. Seeds 42 (all arms) plus 43, 44 for the two main treatments (B, C).
- **Corpus control:** every arm starts from the *same* 1,104,106-line base corpus. Only the
  same 31,860 high-register lines (the FK≥10 slice, ~2.9% of lines) are rewritten; the
  rewrite prompt is the only thing that differs between treatments. Base reconstruction md5
  is identical across arms, so the corpora provably differ only on those lines.
- **Rewriter:** Llama-3.3-70B (via OpenRouter), per-line, meaning preserved.
- **Evaluation:** official BabyLM 2026 strict zero-shot pipeline, `causal` backend.
  Headline metric = **BLiMP**. (Pipeline validated: our re-run of the baseline gives BLiMP
  65.22 vs the published 64.39 — within ~0.8 pt.)

---

## The runs and how the data differs

| Arm | Corpus | What changed in the rewritten lines | Seeds |
|---|---|---|---|
| **A — baseline** | original 10M | nothing (official organizer checkpoint, not retrained) | 42 |
| **B — simplify** | `bb26_simplified` | **syntax + vocabulary**: rewritten as child-directed English — short sentences, simple grammar, common words | 42, 43, 44 |
| **C — register (mild)** | `bb26_register` | **vocabulary only**: formal/Latinate words → plain everyday words; sentence structure and length kept fixed | 42, 43, 44 |
| **D — register (aggressive)** | `bb26_register_strong` | **vocabulary only, pushed hard**: replace *every* formal/Latinate word, prefer phrasal verbs; structure and length still fixed | 42 |

**Manipulation check** — measured on the 31,860 rewritten lines (confirms each treatment did
what it claims). Lower FK grade and lower Latinate % = plainer/more Germanic.

| Metric (changed lines) | A original | B simplify | C register mild | D register aggressive |
|---|---|---|---|---|
| Flesch–Kincaid grade | 14.8 | 2.6 | 14.0 | 12.1 |
| Latinate share (%) | 48.3 | 32.1 | 42.8 | 30.3 |
| Words per line | 54.5 | 59.2 | 54.7 | 55.3 |

The register arms (C, D) form a clean dose-response on vocabulary alone — Latinate share
48.3 → 42.8 → 30.3 — while keeping sentence length essentially flat (structure preserved).
B is the "sledgehammer" that changes both syntax and vocabulary.

---

## Rewrite prompts (verbatim)

Each treatment is defined entirely by its rewrite prompt: the text below was sent as the
**system** message to Llama-3.3-70B, with each corpus line as the user message. These are the
only thing that differs between arms B, C, and D.

**Arm B — `simplify`** (lower register *and* syntax):

> You rewrite text into simple, developmentally-plausible English, the way a caregiver speaks
> to a young child. Use short sentences, common everyday words, and simple grammar; break one
> long sentence into several short ones. Preserve the original meaning and every fact. Do not
> add information, opinions, or commentary, and do not omit facts. Reply with only the
> rewritten text.

**Arm C — `register` (mild)** (vocabulary only, structure fixed):

> You rewrite text by replacing formal, literary, or Latinate words with the plain, everyday
> words a young child would hear (for example: purchase -> buy, comprehend -> understand,
> assist -> help). Keep the sentence's meaning, structure, and length exactly the same: do not
> split, shorten, or reorder sentences, and do not change the grammar. Do not replace words
> that carry specific, technical, or factual meaning (names, places, numbers, and specialist
> terms such as 'toxins' or 'regional'); swap a word only when the plain everyday synonym means
> exactly the same thing. Do not add or remove any facts. Reply with only the rewritten text.

**Arm D — `register_strong` (aggressive)** (vocabulary only, pushed hard):

> You rewrite text into the plainest possible English, using only the simple, everyday words a
> young child would know. Replace every formal, literary, or Latinate word with its most common
> plain equivalent, and prefer short Anglo-Saxon words and phrasal verbs (for example: require
> -> need, obtain -> get, demonstrate -> show, sufficient -> enough, numerous -> many, tolerate
> -> put up with, assist -> help, comprehend -> understand, purchase -> buy, approximately ->
> about, additional -> more). Swap a word even when the plain version is only roughly
> equivalent, as long as the sentence still means the same thing overall. Keep the sentence's
> structure and length the same: do not split, shorten, reorder, or change the grammar. Keep all
> names, places, and numbers exactly as written, and do not add or remove facts. Reply with only
> the rewritten text.

---

## Eval scores (2026 strict zero-shot, accuracy %)

Per-run, all seeds:

| Task | A | B s42 | B s43 | B s44 | C s42 | C s43 | C s44 | D s42 | D s43 | D s44 |
|---|---|---|---|---|---|---|---|---|---|---|
| **BLiMP** | **65.22** | 63.96 | 64.34 | 64.33 | 63.83 | 64.57 | 62.67 | 63.41 | 63.44 | 64.47 |
| BLiMP-supplement | 57.25 | 57.33 | 58.33 | 53.72 | 58.14 | 52.02 | 55.61 | 55.65 | 54.81 | 56.46 |
| COMPS | 51.81 | 51.74 | 51.78 | 51.93 | 51.35 | 51.86 | 51.75 | 51.91 | 52.23 | 51.74 |
| Entity tracking | 21.07 | 14.55 | 20.25 | 19.36 | 39.95 | 29.42 | 12.86 | 19.80 | 31.09 | 22.44 |
| Reading — eye-track | 9.63 | 7.07 | 7.93 | 6.80 | 7.37 | 9.02 | 7.98 | 6.37 | 7.07 | 8.34 |
| Reading — self-paced | 1.64 | 2.30 | 2.36 | 2.14 | 1.74 | 1.85 | 2.43 | 1.69 | 1.16 | 2.65 |

Aggregated (mean ± sd over seeds; A is single-seed):

| Task | A | B simplify | C register mild | D register aggr. |
|---|---|---|---|---|
| **BLiMP** | 65.22 | 64.21 ± 0.22 | 63.69 ± 0.96 | 63.77 ± 0.60 |
| BLiMP-supplement | 57.25 | 56.46 ± 2.42 | 55.26 ± 3.08 | 55.64 ± 0.83 |
| COMPS | 51.81 | 51.82 ± 0.10 | 51.65 ± 0.27 | 51.96 ± 0.25 |
| Entity tracking | 21.07 | 18.05 ± 3.06 | 27.41 ± 13.66 | 24.44 ± 5.91 |
| Reading — eye-track | 9.63 | 7.27 ± 0.59 | 8.12 ± 0.83 | 7.26 ± 1.00 |
| Reading — self-paced | 1.64 | 2.27 ± 0.11 | 2.01 ± 0.37 | 1.83 ± 0.76 |

**Reading** is a *correlation* between model word-surprisal and human reading times (eye-tracking
and self-paced reading), not an accuracy — higher = the model's predictions better track human
reading effort. It is reported on a different scale from the accuracy tasks above. The baseline (A)
tracks human eye-movements best; the treatments are lower, consistent with the BLiMP picture.

## Not yet measured (and why)

The BabyLM leaderboard lists a few more categories we have not run. None change the headline; they
are scoped out for cost or access reasons:

- **EWoK** (world knowledge) — the eval set `ewok-core/ewok-core-1.0` is a **gated** Hugging Face
  dataset and requires a one-time access request on the account before it can be downloaded. Pending
  that approval; the eval itself is cheap (~6 min/arm) and will be added once access is granted.
- **(Super)GLUE** — this is the **fine-tuning** track (train a classifier head per task: BoolQ, MNLI,
  MRPC, …), a separate and much heavier pipeline than the zero-shot suite above. Deferred as
  out-of-scope for the register/grammar question; candidate for a downstream-transfer follow-up.
- **AoA** (Age of Acquisition) — **now MEASURED**; see Finding 5 below. (Retrained A/B/C/D at seed 42
  with 22 milestone checkpoints each; harvested per-word surprisal over training on a fixed probe set
  of 67 matched Latinate/Germanic synonym pairs.)
- **Text Average** — not a measurement; it is just the mean over the text tasks above.

---

## Findings

1. **No improvement from any treatment.** On BLiMP, all treatments land ~1–1.5 pt *below* the
   untouched baseline (A 65.22; B/C/D ≈ 63.4–64.2).

2. **No register dose-response.** Pushing register 3× harder (C −5.5 pp → D −18 pp Latinate)
   moved BLiMP 63.69 ± 0.96 → 63.77 ± 0.60 — no meaningful change; the two are multi-seed and
   their error bars overlap completely. So the null result is not a "didn't push hard enough"
   artifact; lexical register, isolated from syntax, simply does not drive grammatical learning here.

3. **The entity-tracking "win" was noise.** A single seed of Arm C scored 39.95, but across
   seeds it swings 39.95 / 29.42 / 12.86 (± 13.7). Multi-seed replication dissolved it. COMPS
   sits at chance (~51.8) for every arm; supplement is noisy with no clear signal.

4. **Caveat.** Arm A is a single seed (the official checkpoint), so the baseline gap lacks its
   own error bar; a multi-seed, self-trained Arm A on the original corpus is now in training to
   close this. The dose-response conclusion (C vs D, both multi-seed, flat) is already robust.

5. **Acquisition (AoA) tells a different — and positive — story.** We tracked per-word surprisal
   across training for a fixed set of 67 matched Latinate/Germanic synonym pairs, split by etymology.
   The baseline (A) reproduces the human ordering: Germanic core words acquired fast and early, the
   Latinate layer starting far worse and catching up late (gap 6.78 → 2.22 bits over training). The
   register treatments **dose-dependently slow Latinate acquisition** — final Latinate−Germanic gap
   A 2.22 < C 3.14 (−5.5 pp Latinate) < D 3.73 (−18 pp Latinate), monotonic, while Germanic stays flat
   (~8.7–9.0 bits) since those words were never removed. See `figures/register/aoa_trajectory.png`,
   `figures/register/aoa_gap.png`.

**Takeaway:** Lowering register/complexity of the hard slice of the corpus does not buy
sample efficiency for grammar (BLiMP) under a controlled, multi-seed comparison — **but it is not
inert.** The two evaluation axes dissociate cleanly: BLiMP's register dose-response is flat, while
lexical acquisition (AoA) degrades monotonically with dose. The manipulation demonstrably reshapes
what the model learns at the vocabulary level; that change simply does not transfer to grammatical
benefit.
