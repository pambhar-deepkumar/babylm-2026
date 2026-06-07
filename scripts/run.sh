#!/usr/bin/env bash
# run.sh – BabyLM 2026 experiment runner
#
# --language is required. --mode controls hyperparameter presets.
#
# Examples
#   bash run.sh --language strict_small
#   bash run.sh --mode full --language strict_small
#   bash run.sh --mode full --language nld --tokenizer src/babylm_2026/tokenizers/nld_spm.model
#   bash run.sh --mode full --language zho --tokenizer src/babylm_2026/tokenizers/zho_spm.model
#   bash run.sh --mode full --language nld --epochs 20 --tag nld_v2
#   bash run.sh --mode debug --language strict_small --debug   # also subsets data to 100 lines
#
# After training:
#   bash scripts/evaluate.sh src/babylm_2026/models/<tag> mlm

set -euo pipefail

# ─────────────────────────────────────────── editable defaults ──────────────
DEFAULT_BASE_MODEL="microsoft/deberta-v3-base"
DEFAULT_TOKENIZER="src/LukasLM/tokenizers/bb24.model"
TRAIN_SCRIPT="src/LukasLM/train_mask.py"
EVAL_SCRIPT="scripts/evaluate.sh"
# ────────────────────────────────────────────────────────────────────────────

usage() {
  sed -n '2,/^$/s/^# \?//p' "$0"
  exit "${1:-0}"
}

# ── flags ────────────────────────────────────────────────────────────────────
LANGUAGE=""
MODE="debug"
MAX_EPOCHS=""
BATCH_SIZE=""
BASE_MODEL="${DEFAULT_BASE_MODEL}"
LR=""
OUTPUT_TAG=""
TOKENIZER=""
MAX_SEQ_LEN=""
GRAD_ACC=1
DEBUG_FLAG=""       # --debug subsets data to 100 lines (passed to train_mask.py)
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --language)    LANGUAGE="$2";    shift 2 ;;
    --mode)        MODE="$2";        shift 2 ;;
    --epochs)      MAX_EPOCHS="$2";  shift 2 ;;
    --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
    --lr)          LR="$2";          shift 2 ;;
    --base_model)  BASE_MODEL="$2";  shift 2 ;;
    --tokenizer)   TOKENIZER="$2";   shift 2 ;;
    --max_seq_len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --grad_acc)    GRAD_ACC="$2";    shift 2 ;;
    --tag)         OUTPUT_TAG="$2";  shift 2 ;;
    --debug)       DEBUG_FLAG="--debug"; shift ;;
    --help|-h)     usage 0 ;;
    --*)           EXTRA_ARGS+=("$1"); shift ;;
    *)             echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$LANGUAGE" ]]; then
  echo "ERROR: --language is required (strict_small | strict | nld | zho)"
  usage
fi

# ── normalise language and select data folder ────────────────────────────────
DATASET_DIR=""
EVAL_LANG=""
case "${LANGUAGE,,}" in
  strict_small)
    DATASET_DIR="src/babylm_2026/data/strict_small"
    EVAL_LANG="en"
    ;;
  strict)
    DATASET_DIR="src/babylm_2026/data/strict"
    EVAL_LANG="en"
    ;;
  nld|nl)
    LANGUAGE="nld"
    DATASET_DIR="src/babylm_2026/data/nld"
    EVAL_LANG="nld"
    ;;
  zho|zh)
    LANGUAGE="zho"
    DATASET_DIR="src/babylm_2026/data/zho"
    EVAL_LANG="zho"
    ;;
  *)
    echo "Unknown language: $LANGUAGE  (use strict_small | strict | nld | zho)"
    exit 1
    ;;
esac

mkdir -p "$DATASET_DIR"

# ── pick training file ───────────────────────────────────────────────────────
TRAIN_FILE=""
for candidate in \
  "${DATASET_DIR}/texts1.txt" \
  "${DATASET_DIR}/texts2.txt" \
  "${DATASET_DIR}/train.txt"
do
  if [[ -f "$candidate" ]]; then
    TRAIN_FILE="$candidate"
    break
  fi
done

# fallback: any *.train.txt in the folder
if [[ -z "$TRAIN_FILE" ]]; then
  TRAIN_FILE="$(find "$DATASET_DIR" -maxdepth 1 -name '*.train.txt' | head -n 1)"
fi

if [[ -z "$TRAIN_FILE" ]]; then
  echo "ERROR: Cannot find a training file in $DATASET_DIR"
  echo "Expected one of: texts1.txt, texts2.txt, train.txt, or *.train.txt"
  ls -la "$DATASET_DIR" 2>/dev/null || true
  exit 1
fi

# ── pick validation file ─────────────────────────────────────────────────────
# Search only within the correct language's folder — never cross-language fallbacks
VALID_FILE=""
for candidate in \
  "${DATASET_DIR}/even.dev.txt" \
  "${DATASET_DIR}/dev.txt" \
  "${DATASET_DIR}/validation.txt" \
  "${DATASET_DIR}/valid.txt"
do
  if [[ -f "$candidate" ]]; then
    VALID_FILE="$candidate"
    break
  fi
done

# fallback: any *.dev.txt or *.valid.txt in the folder
if [[ -z "$VALID_FILE" ]]; then
  VALID_FILE="$(find "$DATASET_DIR" -maxdepth 1 \( -name '*.dev.txt' -o -name '*.valid.txt' \) | head -n 1)"
fi

# last resort: carve 500 lines off the training file
if [[ -z "$VALID_FILE" ]]; then
  VALID_FILE="$(mktemp /tmp/babylm_valid_${LANGUAGE}_XXXXXX.txt)"
  head -n 500 "$TRAIN_FILE" > "$VALID_FILE"
  echo "WARNING: No validation file found. Using first 500 lines of training data as validation."
fi

# ── mode presets ─────────────────────────────────────────────────────────────
case "$MODE" in
  debug)
    MAX_EPOCHS="${MAX_EPOCHS:-1}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    LR="${LR:-5e-4}"
    MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"
    ;;
  full)
    MAX_EPOCHS="${MAX_EPOCHS:-10}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    LR="${LR:-5e-5}"
    MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
    ;;
  *) echo "Unknown mode: $MODE  (use debug | full)"; exit 1 ;;
esac

# ── resolve tokenizer ─────────────────────────────────────────────────────────
# train_mask.py loads the tokenizer with: DebertaV2Tokenizer(args.tokenizer)
# This expects a path directly to a SentencePiece .model file — NOT a folder.
# So --tokenizer should always point to a .model file.
#
# Priority:
#   1. Explicit --tokenizer flag
#   2. tokenizer.model inside the language data folder
#   3. LukasLM's English bb24.model (warns if used for nld/zho)
RESOLVED_TOKENIZER=""

if [[ -n "$TOKENIZER" ]]; then
  if [[ -f "$TOKENIZER" ]]; then
    RESOLVED_TOKENIZER="$TOKENIZER"
  else
    echo "ERROR: Tokenizer file not found: $TOKENIZER"
    exit 1
  fi
elif [[ -f "${DATASET_DIR}/tokenizer.model" ]]; then
  RESOLVED_TOKENIZER="${DATASET_DIR}/tokenizer.model"
elif [[ -f "${DEFAULT_TOKENIZER}" ]]; then
  RESOLVED_TOKENIZER="${DEFAULT_TOKENIZER}"
  if [[ "$LANGUAGE" == "nld" || "$LANGUAGE" == "zho" ]]; then
    echo "WARNING: No language-specific tokenizer found for '${LANGUAGE}'."
    echo "         Falling back to the English bb24.model — cross-lingual results will be poor."
    echo "         Train one with scripts/train_tokenizer.py and pass it with --tokenizer <path.model>"
  fi
else
  echo "ERROR: Cannot resolve a tokenizer."
  echo "       For nld/zho: train with scripts/train_tokenizer.py then pass --tokenizer <path.model>"
  echo "       For strict/strict_small: ensure src/LukasLM/tokenizers/bb24.model exists"
  exit 1
fi

# ── tag and output dir ────────────────────────────────────────────────────────
OUTPUT_TAG="${OUTPUT_TAG:-${LANGUAGE}_${MODE}}"
OUTPUT_DIR="src/babylm_2026/models/${OUTPUT_TAG}"
mkdir -p "$OUTPUT_DIR"

# ── summary ───────────────────────────────────────────────────────────────────
cat <<EOF
══════════════════════════════════════════════════════════════════
  Experiment  : ${OUTPUT_TAG}
  Language    : ${LANGUAGE}
  Eval lang   : ${EVAL_LANG}
  Mode        : ${MODE}
  Train file  : ${TRAIN_FILE}
  Valid file  : ${VALID_FILE}
  Base model  : ${BASE_MODEL}
  Tokenizer   : ${RESOLVED_TOKENIZER}
  Epochs      : ${MAX_EPOCHS}
  Batch size  : ${BATCH_SIZE}  (grad_acc ${GRAD_ACC})
  LR          : ${LR}
  Max seq len : ${MAX_SEQ_LEN}
  Output dir  : ${OUTPUT_DIR}
  --debug     : ${DEBUG_FLAG:-off}
══════════════════════════════════════════════════════════════════
EOF

# ── build python command ──────────────────────────────────────────────────────
PY_CMD=(
  python "${TRAIN_SCRIPT}"
  --train_data    "${TRAIN_FILE}"
  --valid_data    "${VALID_FILE}"
  --output_path   "${OUTPUT_DIR}"
  --model_path    "${BASE_MODEL}"
  --tokenizer     "${RESOLVED_TOKENIZER}"
  --epochs        "${MAX_EPOCHS}"
  --batch_size    "${BATCH_SIZE}"
  --grad_acc      "${GRAD_ACC}"
  --lr            "${LR}"
  --max_seq_len   "${MAX_SEQ_LEN}"
  --seed          42
)

if [[ -n "$DEBUG_FLAG" ]]; then
  PY_CMD+=("--debug")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  PY_CMD+=("${EXTRA_ARGS[@]}")
fi

# save exact command for reproducibility
printf '%s\n' "${PY_CMD[@]}" > "${OUTPUT_DIR}/last_cmd.txt"

# ── run ───────────────────────────────────────────────────────────────────────
"${PY_CMD[@]}"

echo
echo "Training complete.  Weights → ${OUTPUT_DIR}/"
echo "Evaluate with:  bash ${EVAL_SCRIPT} ${OUTPUT_DIR} mlm"