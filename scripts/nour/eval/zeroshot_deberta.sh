#!/bin/bash
# Zero-shot evaluation for encoder-only DeBERTa masked-LM checkpoints.
# Uses pseudo-log-likelihood scoring (Salazar et al. 2020) via deberta_mlm_model.py.
#
# Usage:
#   bash zeroshot_deberta.sh --model_path /abs/path/to/checkpoint \
#       [--langs "eng nld zho"] [--batch_size 8] [--max_length 256] \
#       [--preseg_model /abs/path/to/bpe.model]
#
# --preseg_model: path to a per-language SentencePiece BPE model used for
#   two-stage tokenization (needed for meta-tokenizer checkpoints such as
#   multi_run2 and multi_run3). Raw text is first segmented with this BPE
#   model, then encoded by the checkpoint's word-type meta-tokenizer.

model_path=""
langs="eng nld zho"
batch_size=8
max_length=256
preseg_model=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)   model_path="$2";   shift 2 ;;
        --langs)        langs="$2";        shift 2 ;;
        --batch_size)   batch_size="$2";   shift 2 ;;
        --max_length)   max_length="$2";   shift 2 ;;
        --preseg_model) preseg_model="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$model_path" ]]; then
    echo "Error: --model_path is required"; exit 1
fi

# Derive a short name for the output directory from the checkpoint path
run_name=$(basename "$(dirname "$model_path")")__$(basename "$model_path")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MULTILINGUAL_DIR="$(dirname "$SCRIPT_DIR")"

# Build model_args string — include preseg_model only when specified
if [[ -n "$preseg_model" ]]; then
    model_args="pretrained=${model_path},max_length=${max_length},preseg_model=${preseg_model}"
else
    model_args="pretrained=${model_path},max_length=${max_length}"
fi

for lang in $langs; do
    task_name="zeroshot_${lang}"
    echo "=== Evaluating ${lang} for ${run_name} ==="
    python3 "${MULTILINGUAL_DIR}/run_deberta_eval.py" \
        --model deberta-mlm \
        --model_args "${model_args}" \
        --tasks "${task_name}" \
        --device cuda \
        --batch_size ${batch_size} \
        --output_path "${MULTILINGUAL_DIR}/results/nour/${run_name}" \
        --num_fewshot 0 \
        --log_samples \
        --include_path "${MULTILINGUAL_DIR}/tasks/" \
        2>&1
    echo "=== Done: ${lang} ==="
done

