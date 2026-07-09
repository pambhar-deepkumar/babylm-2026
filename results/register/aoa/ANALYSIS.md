# Age-of-Acquisition (AoA) Analysis — Register Arms

**One line:** Lowering the corpus register does nothing to grammar (BLiMP, flat) but
**dose-dependently slows how well the model learns Latinate vocabulary** — the two evaluation
axes dissociate.

## Method

- **Probe set:** 67 curated (Latinate, Germanic) synonym pairs → 123 words (e.g. *acquire/get,
  assist/help, purchase/buy, comprehend/understand*). For each word, up to 30 natural contexts
  sampled from the original corpus `bb26_en.train` → 3,605 fixed probes, **identical across all
  arms** (only the trained model differs).
- **Signal:** per-word **surprisal** = −log₂ p(word tokens | left context), in bits. Lower = better
  acquired. Aggregated to a class mean (Germanic vs Latinate) at each checkpoint.
- **Checkpoints:** each arm (A/B/C/D) retrained from scratch at **seed 42** with 22 milestone
  checkpoints on a matched token grid (1M … 250M tokens seen). Harvested per-word surprisal at each.
- **Arms:** A = original corpus; C = register-mild (−5.5pp Latinate, vocab only); D = register-
  aggressive (−18pp Latinate, vocab only); B = simplify (syntax + vocab). A→C→D is the vocab-only
  dose ladder.

## Result

**Baseline (A) reproduces the human ordering:** Germanic core words acquired fast and early;
the Latinate layer starts far worse and catches up late (gap closes 6.78 → 2.22 bits over training).

**The register treatments dose-dependently slow Latinate acquisition** (final = 250M tokens seen):

| Arm | Latinate removed | Latinate surprisal | Germanic surprisal | **Lat−Ger gap** |
|---|---|---|---|---|
| A original | — | 10.91 | 8.69 | **2.22** |
| C register mild | −5.5pp | 11.87 | 8.73 | **3.14** |
| D register aggressive | −18pp | 12.72 | 8.99 | **3.73** |
| B simplify (syntax+vocab) | — | 12.62 | 9.40 | 3.22 |

- Gap **A 2.22 < C 3.14 < D 3.73** — monotonic, and the ordering holds across the trajectory
  (see `aoa_gap.png`).
- **Germanic stays flat** (~8.7–9.0 bits across arms): the treatment hits only the layer it removes.
- B sits between C and D, consistent with it being a stronger (syntax + vocab) manipulation.

## The headline: the two axes dissociate

| | Register dose-response |
|---|---|
| **BLiMP** (grammar) | **flat** — A/C/D all ~63.4–63.8, no trend |
| **AoA** (lexical acquisition) | **monotonic** — gap 2.22 → 3.14 → 3.73 |

The manipulation demonstrably and dose-dependently reshapes what the model learns at the vocabulary
level; that change simply does not transfer to grammatical benefit. A null benchmark result becomes
a real, mechanistic, dose-dependent finding.

## Files

- `traj_{a,b,c,d}_surprisal.csv` — tidy per-(checkpoint, word) surprisal (columns: arm, step,
  tokens_seen, word, klass, pair_id, surprisal_bits, n).
- `aoa_trajectory.png/svg` — Latinate surprisal per arm vs tokens seen (the dose ladder).
- `aoa_gap.png/svg` — Latinate−Germanic gap per arm vs tokens seen (monotonic A<C<D).

## Reproduce

```
# 1. build the fixed probe set from the original corpus
python scripts/register/build_aoa_probes.py --corpus data/bb26_en.train --k 30
# 2. harvest surprisal over an arm's 22 checkpoints (GPU)
sbatch --export=ALL,ARM_DIR=output/traj_a,ARM_NAME=A,OUT=results/register/aoa/traj_a_surprisal.csv scripts/register/harvest_traj.slurm
# 3. plot
python scripts/register/plot_aoa.py --indir results/register/aoa --outdir figures/register
```
