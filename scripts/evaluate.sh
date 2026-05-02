#!/usr/bin/env bash
# Wrapper around the BabyLM 2026 strict-small evaluation pipeline.
# Usage: scripts/evaluate.sh <model_name_or_path> [backend]
#   model_name_or_path  HuggingFace Hub model ID or local checkpoint path
#   backend             mlm|causal|mntp|enc_dec_mask|enc_dec_prefix (default: mlm)

set -euo pipefail

MODEL="${1:?model name or path required}"
BACKEND="${2:-mlm}"
TRACK="strict-small"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$REPO_ROOT/babylm-eval/strict"

if [ ! -d "$EVAL_DIR/scripts" ]; then
  echo "Eval submodule missing at $EVAL_DIR — run: make setup" >&2
  exit 1
fi

cd "$EVAL_DIR"
bash scripts/collate_preds.sh "$MODEL" "$BACKEND" "$TRACK"
