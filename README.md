# BabyLM 2026 — TUM LLM Practical Course

Submission for the [4th BabyLM Challenge](https://babylm.github.io/) at EMNLP 2026.

**Team:** Deep Pambhar, Alessio Piroli, Nour Hadjfredj
**TA:** Lukas
**Track (tentative):** Multilingual (English / Dutch / Chinese)

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
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Add dependencies to `pyproject.toml` as the project grows.

## Upstream BabyLM repos (clone as siblings or git submodules)

| Repo | Purpose |
|---|---|
| [babylm/evaluation-pipeline-2025](https://github.com/babylm/evaluation-pipeline-2025) | Closest to the 2026 eval pipeline (2026 repo not yet released as of 2026-05-02). |
| [babylm/baseline-pretraining](https://github.com/babylm/baseline-pretraining) | GPT-BERT and GPT-2 Small baseline training code. |
| [babylm/babylm_data_preprocessing](https://github.com/babylm/babylm_data_preprocessing) | Official corpus preprocessing. |

The 2026 evaluation pipeline and detoxified corpus drop "early April 2026" per
the CFP — check the BabyLM Slack and GitHub.

## Key dates

- **2026-05-25:** ARR submission deadline (skipping — too tight).
- **Mid July 2026:** Direct OpenReview submission deadline (our target).
- **Mid August 2026:** Decisions released.
- **Early September 2026:** Camera-ready.
- **24–29 October 2026:** Workshop @ EMNLP Budapest.

## Track + research questions

Working plan, to be confirmed with the TA:

- **Track:** Multilingual (English / Dutch / Chinese).
- **RQ1:** Ablate the EN/NL/ZH language ratio under a fixed 100M-token
  Byte-Premium-adjusted budget; measure cross-lingual transfer to each
  language's eval suite.
- **RQ2:** Ablate the causal:masked ratio of GPT-BERT (2025 baseline ships
  at roughly 1:7) in the multilingual setting.
