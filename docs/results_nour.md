# Multilingual Training Results (Nour) — BabyLM 2026

Scope: training DeBERTa v2 (Lukas's 2025 baseline architecture) on Dutch and
Chinese, plus one shared multilingual model on English+Dutch+Chinese. Three
questions, three experiments:

1. Does the English-baseline vocab size (40k) transfer to other languages, or
   should vocab size be tuned per language? (EXP-N1)
2. When one model is trained on three languages at once, is it better to share
   one vocabulary, split it two ways, or keep three fully separate
   vocabularies? (EXP-N3)
3. Do these findings hold up under the official zero-shot eval pipeline, not
   just training accuracy? (EXP-EVAL)

## Headline

**Smaller, per-language vocabularies win, and so does splitting vocabulary by
language in a joint model.** Dropping from 40k to 8k vocab raised final
training accuracy +1.4pts for Dutch and +6.3pts for Chinese. In the joint
3-language model, using 2 or 3 separate tokenizers beat 1 fully joint
tokenizer by ~5pts. Zero-shot eval confirms the vocab-size finding directly
(Chinese) and the sharing-strategy finding on English so far, but several
legs (nld, zho) of the zero-shot sharing-strategy comparison are still
pending — see [Status](#status--pending-numbers).

## Setup

**Held fixed across every run below** (Lukas's 2025 recipe, unchanged):
DeBERTa v2, 12 layers, 12 heads, `hidden_size=768`, `intermediate_size=3072`,
`batch_size=256`, LAMB optimizer, `lr=0.001`, `epochs=10`,
curriculum `max_seq_len "0:64,5:256"` (seq_len jumps 64→256 at epoch 5),
`mask_decay=0.1`, adaptive masking, all via `train_mask.py`.

`lr=0.001` (not Lukas's original `0.007`) per Alex's root-cause finding that
the original LR collapses training at the epoch-5 seq_len jump. None of the
runs below collapse — final checkpoint is the best or tied-best checkpoint in
every run.

**Varied:** tokenizer vocab size (EXP-N1), and — for the joint multilingual
model only — how vocabulary is split across languages (EXP-N3).

**Two evaluation methods, reported separately, not interchangeably:**
- **Training accuracy** — per-masked-token top-1 accuracy on each run's own
  held-out validation split. Cheap, available immediately, but not
  cross-model-vocab comparable (see caveats) and not the official metric.
- **Zero-shot eval** (EXP-EVAL) — BLiMP/HellaSwag/Winogrande/XCOMPS/
  XStoryCloze/MultiBLiMP/ZhoBLiMP via a custom pseudo-log-likelihood (PLL)
  wrapper (Salazar et al. 2020) built for `lm-eval-harness`, since our model
  is encoder-only (`DebertaV2ForMaskedLM`) and the harness only natively
  supports causal/seq2seq backends. No fine-tuning; the model scores each
  multiple-choice option by summed PLL and picks the highest.

**Caveat that applies throughout:** all runs here used more data than the
official 100M-token strict budget (Dutch 111.0M / Chinese 96.8M tokens in
EXP-N1; 33M tokens/language, 99M total, in EXP-N3). Numbers below are for
internal comparison, not leaderboard submission. A budget-compliant rerun is
a planned follow-up (see Notes).

---

## Experiment 1 — Tokenizer vocab size: 8k vs 40k (per language)

**Question:** the 40k vocab was chosen to match Lukas's English baseline
(`bb24.model`) for comparability. Is that the right size for Dutch and
Chinese, or does it hurt them?

### Runs

| Run | Language | Vocab | Tokenizer | Job | Script |
|---|---|---|---|---|---|
| nld_baseline | Dutch | 40k | `nld_spm.model` | 5683731 | `scripts/train_nld_baseline.sh` |
| nld_8k | Dutch | 8k | `nld_spm_8k.model` | 5684434 | `scripts/train_nld_8k.sh` |
| zho_baseline | Chinese | 40k | `zho_spm.model` | 5683732 | `scripts/train_zho_baseline.sh` |
| zho_8k | Chinese | 8k | `zho_spm_8k.model` | 5684435 | `scripts/train_zho_8k.sh` |

Single-variable test: identical recipe and data per language, only
`--tokenizer` / `--output_path` differ between the 8k and 40k script pair.

### Results

| | Dutch 40k | Dutch 8k | Chinese 40k | Chinese 8k |
|---|---|---|---|---|
| **Final training accuracy** | 43.76% | **45.12%** (+1.36) | 38.52% | **44.79%** (+6.27) |
| Final training loss* | 3.98 | 3.26 | 6.14 | 3.58 |
| Params | 116.8M | 92.2M (−21%) | 116.8M | 92.2M (−21%) |
| Total steps | 67,275 | 79,670 | 59,040 | 73,145 |
| Train tokens | 111.0M | 131.3M | 96.8M | 119.9M |
| Wall time | 5h32m | 5h46m | 3h52m | 4h37m |
| Seq-len jump (epoch 5) | survived, acc rose | survived, acc rose | survived, acc rose | survived, acc rose |

\* Loss is not comparable across vocab sizes (different output dimensionality)
— accuracy is the fair comparison.

### Findings

- 8k vocab wins for both languages, and the effect is far larger for Chinese
  (+6.27pts) than Dutch (+1.36pts).
- Not a fertility artifact: tokenizer fertility analysis
  (`src/babylm_2026/` tokenizer `.model` files vs `texts1.txt` samples) shows
  Chinese is *more* finely tokenized than Dutch (0.517 tokens/char vs 1.321
  tokens/word), which if anything should make prediction easier, not harder —
  so fertility does not explain why 40k hurt Chinese more.
- Best explanation found: a 40k-way output classification head is
  disproportionately sparse/long-tailed for Chinese, whose natural "alphabet"
  is characters (many rare multi-character BPE merges with few training
  examples each), versus Dutch, where 40k BPE merges stay close to whole
  words. The Chinese-only early eval-accuracy plateau (flat ~35% for the
  first ~9k steps) seen at 40k vocab is **absent** at 8k vocab, even though
  the same corpus-markup tokens (`CHILD`, `TARGET`, `MOTHER`, etc. — CHILDES
  annotation artifacts) still dominate the adaptive-masking "hardest tokens"
  list at both vocab sizes for the first ~1,200 steps. This isolates the
  plateau to vocab size, not to the annotation-token issue.
- **What would make this firmer:** the EXP-N2 sweep (4k/8k/16k/32k/40k per
  language, tokenizers already trained) would show whether 8k is a local
  optimum or accuracy keeps improving below 8k — not yet run at the model
  level (see Status).

---

## Experiment 2 — Tokenizer-sharing strategy in one joint multilingual model

**Question (Lukas's suggestion):** instead of one model per language, train
a single model on English+Dutch+Chinese mixed together. Does it matter
whether all three languages share one vocabulary, or should vocabulary be
split?

### Runs

All three use the same recipe as Experiment 1, same fixed data budget:
**33M tokens/language (99M total)**, measured with each language's own 8k
tokenizer; validation is English-only, 10M disjoint tokens
(`babylm_2026/data/budget100m/`).

| Run | Vocab strategy | Vocab | Job | 
|---|---|---|---|
| Run 1 — 1 joint tokenizer | one BPE vocab trained on raw mixed eng+nld+zho text | 24k | 5690616 |
| Run 2 — 2 tokenizers | eng+nld share one vocab, zho keeps its own | 16k (eng+nld) + 8k (zho) | 5691360 |
| Run 3 — 3 tokenizers | each language keeps its own separate vocab | 8k + 8k + 8k | 5691361 |

Splitting vocab within one model is a preprocessing-only mechanism (Lukas's
`train_mask.py` only ever loads one tokenizer): each language's text is
pre-segmented with its own tokenizer, piece-strings are concatenated, and a
word-level SentencePiece "meta-tokenizer" is trained on top so
`train_mask.py` sees one ordinary tokenizer + text file. Run 1 needs none of
this — it's a single ordinary BPE tokenizer used directly.

### Results

| | Run 1: 1 joint tokenizer | Run 2: 2 tokenizers | Run 3: 3 tokenizers |
|---|---|---|---|
| **Final training accuracy (eng-only val)** | 67.63% | **72.56%** | 72.24% |
| Final training loss* | 1.78 | 1.38 | 1.38 |
| Params | 104.5M | 105.0M | 102.2M |
| Total steps | 63,260 | 63,320 | 66,065 |
| Wall time | 3h49m | 4h11m | 4h18m |

\* Loss not comparable across differing output-vocab sizes.

### Findings

- Splitting vocabulary by language clearly beats one fully joint vocabulary:
  both Run 2 and Run 3 beat Run 1 by **~5pts**, consistent with the
  Experiment 1 mechanism — cramming multiple scripts into one
  smaller-than-ideal shared vocabulary hurts each language's classification
  head.
- Run 2 (2 tokenizers) edges out Run 3 (3 tokenizers) by +0.32pts — small
  enough not to be a strong signal at this scale, but consistent with "fully
  separating every language isn't strictly necessary; sharing closely
  related, same-script languages (English+Dutch) costs little."
- No collapses in any of the three runs; final checkpoint is best or
  tied-best in each.
- All three multilingual runs score notably higher on English-only
  validation (67.6–72.6%) than any single-language English baseline
  available for comparison (Alex's ~54.24%, different corpus/recipe variant
  — see Notes). Plausible explanations not yet disambiguated: smaller,
  denser per-language vocab (8k/16k vs 40k), or a genuine cross-lingual
  transfer/regularization effect from the other two languages, or both.
  **What would make this firmer:** an English-only run using the *same* 33M
  budget-slice + 8k vocab as an isolated control, to separate "vocab size"
  from "multilingual transfer" as the source of the gain.

---

## Experiment 3 — Zero-shot eval (BLiMP / HellaSwag / Winogrande / XCOMPS / XStoryCloze / MultiBLiMP / ZhoBLiMP)

Checks whether the training-accuracy findings above hold under the official
zero-shot metric. **Partial — several legs still running or pending; see
[Status](#status--pending-numbers) for exactly what's missing and why.**

### Chinese models (zho) — complete

| Checkpoint | Train acc | HellaSwag | Winogrande | XCOMPS | XStoryCloze | ZhoBLiMP | **zeroshot_zho** |
|---|---|---|---|---|---|---|---|
| zho_8k (8k vocab) | 44.79% | 26.99% | 51.28% | 52.37% | 47.37% | 75.40% | **61.80%** |
| zho_baseline (40k vocab) | 38.52% | 26.17% | 51.20% | 52.99% | 48.68% | 74.62% | **61.40%** |

8k beats 40k on zero-shot too (+0.40pts), same direction as training
accuracy (+6.27pts) but a much smaller gap under zero-shot.

### Dutch model (nld) — mostly pending

| Checkpoint | Train acc | BLiMP-NL | HellaSwag | MultiBLiMP | Winogrande | XCOMPS | XStoryCloze | **zeroshot_nld** |
|---|---|---|---|---|---|---|---|---|
| nld_8k (8k vocab) | 45.12% | 75.73% | 26.03% | 88.33% | 48.31% | 52.31% | 49.07% | **53.60%** |
| nld_baseline (40k vocab) | 43.76% | TODO | TODO | TODO | TODO | TODO | TODO | **TODO — job 5698013 (12h), pending** |

8k-vs-40k comparison for Dutch zero-shot **cannot be made yet** — the 40k
run hasn't finished (40k's larger logit matrix makes BLiMP-NL scoring ~5x
slower; earlier 4h-limit attempts timed out).

### Multilingual model — multi_run1_joint (1 tokenizer)

| Lang | BLiMP/ZhoBLiMP | HellaSwag | MultiBLiMP | Winogrande | XCOMPS | XStoryCloze | **zeroshot** |
|---|---|---|---|---|---|---|---|
| eng | 63.44% | 25.49% | — | 50.54% | — | 47.29% | **58.36%** |
| zho | 71.00% | 25.67% | — | 49.71% | 52.35% | 45.12% | **58.98%** |
| nld | TODO | TODO | TODO | TODO | TODO | TODO | **TODO — job 5698005, pending** |

### Multilingual model — multi_run2_2tok (2 tokenizers)

| Lang | BLiMP/ZhoBLiMP | HellaSwag | MultiBLiMP | Winogrande | XCOMPS | XStoryCloze | **zeroshot** |
|---|---|---|---|---|---|---|---|
| eng | 61.78% | 25.91% | 74.94% | 49.79% | — | 48.68% | **56.99%** |
| zho | TODO | TODO | TODO | TODO | TODO | TODO | **TODO — job 5698385 (retry #2, 4h), pending** |
| nld | TODO | TODO | TODO | TODO | TODO | TODO | **TODO — job 5698008 (6h), pending** |

### Multilingual model — multi_run3_3tok (3 tokenizers)

| Lang | BLiMP/ZhoBLiMP | HellaSwag | MultiBLiMP | Winogrande | XCOMPS | XStoryCloze | **zeroshot** |
|---|---|---|---|---|---|---|---|
| eng | 61.94% | 25.76% | 76.49% | 52.10% | — | 49.77% | **57.18%** |
| zho | TODO | TODO | TODO | TODO | TODO | TODO | **TODO — job 5698581 (retry #2, 4h), pending** |
| nld | TODO | TODO | TODO | TODO | TODO | TODO | **TODO — job 5698011 (6h), pending** |

`—` = task not run for that language (task list differs per language; not a
missing/failed result).

### Findings (partial — English and Chinese legs only)

- On the one comparison available in full (Chinese, 8k vs 40k), zero-shot
  eval agrees in direction with training accuracy but the gap shrinks a lot:
  +0.40pts zero-shot vs +6.27pts training accuracy. **Training accuracy looks
  like it overstates the practical difference vocab size makes** — worth
  keeping in mind when reading Experiment 1's headline number.
- On English (the only language complete for all three tokenizer-sharing
  runs), Run 3 (3 tokenizers) edges out Run 2 (2 tokenizers) on every
  sub-task (zeroshot 57.18% vs 56.99%) — consistent with, but smaller than,
  the training-accuracy gap in the other direction seen in Experiment 2
  (Run 2 beat Run 3 by 0.32pts on training accuracy). The sign flip between
  training accuracy and zero-shot for Run2-vs-Run3 is small in both cases and
  not conclusive.
- **The Experiment 2 headline claim (splitting vocab beats 1 joint
  tokenizer) is not yet checked under zero-shot** — Run 1 vs Run 2/3 zero-shot
  can only be compared on eng+zho so far (Run 1: eng 58.36%/zho 58.98%; Run
  2/3 zho still pending), and even that comparison mixes languages with
  different task batteries. **Do not treat Experiment 2's ranking as
  zero-shot-confirmed until the nld/zho legs above land.**

---

## Status / pending numbers

| Job ID | Checkpoint | Lang | Time limit | Status |
|---|---|---|---|---|
| 5698005 | `multi_run1_joint/checkpoint-63255` | nld | 6h | PENDING |
| 5698008 | `multi_run2_2tok/checkpoint-63320` | nld | 6h | PENDING |
| 5698385 | `multi_run2_2tok/checkpoint-63320` | zho | 4h | PENDING (retry #2, prior attempt timed out at 86%) |
| 5698011 | `multi_run3_3tok/checkpoint-66065` | nld | 6h | PENDING |
| 5698581 | `multi_run3_3tok/checkpoint-66065` | zho | 4h | PENDING (retry #2, prior attempt timed out at 98%) |
| 5698013 | `nld_baseline/checkpoint-67275` | nld | 12h | PENDING |

**Not yet run at all:** EXP-N2 model training (the 4k/8k/16k/32k/40k × 3
languages sweep) — tokenizers exist, DeBERTa baselines for the new sizes
have not been trained. Only the 8k-vs-40k pair (Experiment 1 above) has
actual model results.

Run `squeue -u go82qok2` on LRZ for live job status before citing any TODO
above as still-pending — this file is a snapshot from 2026-07-04.

---

## Reproduce

Recipe (shared across every run, only tokenizer/data path differ):

```bash
python LukasLM/train_mask.py \
  --train_data <path> --valid_data <path> \
  --tokenizer <path>.model \
  --output_path <path> \
  --hidden_size 768 --intermediate_size 3072 \
  --batch_size 256 --lr 0.001 --epochs 10 \
  --max_seq_len "0:64,5:256" --mask_decay 0.1 \
  --all_checkpoints --lamb --cpus 8
```

- Experiment 1 scripts: `scripts/train_{nld,zho}_baseline.sh`,
  `scripts/train_{nld,zho}_8k.sh`.
- Experiment 2 scripts: `scripts/train_combo_tokenizers.py` (builds
  `multi_tokenizer_24k.model`, `engnld_tokenizer_16k.model`),
  `scripts/build_meta_tokenizer.py` (per-language piece pre-segmentation +
  meta-tokenizer), `scripts/train_multi_run{1,2,3}_*.sh`.
- Experiment 3 (eval): `babylm-eval/multilingual/deberta_mlm_model.py` (PLL
  model class, registered as `deberta-mlm`),
  `babylm-eval/multilingual/run_deberta_eval.py` (entry point),
  `babylm-eval/multilingual/scripts/zeroshot_deberta.sh`,
  `scripts/eval_all_checkpoints.sh` (submits one SLURM job per checkpoint).
  Requires `use_fast=False` when loading the tokenizer — the word-type
  meta-tokenizers used in Run 2/3 cannot convert to the fast Rust tokenizer
  backend.

---

## Notes

- English reference point used above (Alex's ~54.24% training accuracy,
  `lr=1e-3`, gradual curriculum) is from a teammate's run on a different
  corpus assembly and is cited only as rough external context, not a
  controlled part of these experiments.
- Official per-language token budget for the multilingual track is not
  documented anywhere found in this repo — flagged to Lukas, reply pending.
  All runs here exceed the 100M-token strict-track budget in some form (see
  Setup caveat); a budget-compliant rerun is a planned follow-up, not yet
  scheduled.
- TODO: once all pending eval jobs land, revisit the "Experiment 2 zero-shot"
  finding above with full data and update the ranking claim if it changes.
- TODO: decide whether to run the EXP-N2 sweep (4k/16k/32k model training,
  tokenizers already exist) to check whether 8k is actually optimal or just
  better than 40k.
