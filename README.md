# BabyLM 2026 — TUM LLM Practical Course

Submission for the [4th BabyLM Challenge](https://babylm.github.io/) at EMNLP 2026.

**Team:** Deep Pambhar, Alessio Piroli, Nour Hadjfredj
**TA:** Lukas
**Track:** Strict-Small (English, 10M words, 2026 detoxified corpus)

## Layout

```
src/babylm_2026/   Python package — training, eval, data loading
experiments/       Experiment configs and runnable scripts (one folder per RQ)
notebooks/         Exploratory analysis
paper/             Workshop paper (LaTeX or markdown drafts)
```

`data/`, `models/`, `checkpoints/`, and `results/raw/` are gitignored — keep
large artefacts off the repo and on HuggingFace Hub or local scratch.

## Setup

```bash
git clone --recursive git@github.com:pambhar-deepkumar/babylm-2026.git
cd babylm-2026
python -m venv .venv && source .venv/bin/activate
make setup
```

`make setup` initialises the eval submodule and installs all
requirements. If you forgot `--recursive`, just run `make setup` — it
handles the submodule.

## Evaluation

The official BabyLM 2026 evaluation pipeline is vendored as a git
submodule at `third_party/babylm-eval/`, pinned to a specific commit so
everyone runs the same eval. See [`docs/eval.md`](docs/eval.md) for how
to score a checkpoint and how to bump the pinned version.

## Key dates

- **2026-05-25:** ARR submission deadline (skipping — too tight).
- **Mid July 2026:** Direct OpenReview submission deadline (our target).
- **Mid August 2026:** Decisions released.
- **Early September 2026:** Camera-ready.
- **24–29 October 2026:** Workshop @ EMNLP Budapest.

## Track + research questions

- **Track:** Strict-Small (English, 10M words, 2026 detoxified corpus).
- **Research questions:** TBD — to be agreed with the TA before the
  first baseline is locked in.
