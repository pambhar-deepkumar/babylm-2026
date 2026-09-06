#!/bin/bash
# Submit zero-shot evaluation jobs for all of Nour's best checkpoints.
# Each job runs on one H100 GPU. The deberta_mlm_model.py wrapper must be
# importable from the working directory (it's in babylm-eval/multilingual/).
#
# Run from: ~/git/nour/  on LRZ
# Usage: bash scripts/eval_all_checkpoints.sh

BASE_MODELS=~/git/nour/babylm_2026/models
EVAL_DIR=~/git/nour/babylm-eval/multilingual
ENV=~/babylm_env

# Map: checkpoint_dir  ->  languages to evaluate
declare -A CHECKPOINTS
CHECKPOINTS["nld_baseline/checkpoint-67275"]="nld"
CHECKPOINTS["zho_baseline/checkpoint-59040"]="zho"
CHECKPOINTS["nld_8k/checkpoint-71703"]="nld"
CHECKPOINTS["zho_8k/checkpoint-73145"]="zho"
CHECKPOINTS["multi_run1_joint/checkpoint-63255"]="eng nld zho"
CHECKPOINTS["multi_run2_2tok/checkpoint-63320"]="eng nld zho"
CHECKPOINTS["multi_run3_3tok/checkpoint-66065"]="eng nld zho"

for CKPT_REL in "${!CHECKPOINTS[@]}"; do
    LANGS="${CHECKPOINTS[$CKPT_REL]}"
    CKPT_ABS="${BASE_MODELS}/${CKPT_REL}"
    RUN_NAME=$(echo "$CKPT_REL" | tr '/' '_')

    JOB_SCRIPT=$(mktemp /tmp/eval_${RUN_NAME}_XXXX.sh)
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=eval_${RUN_NAME}
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=${BASE_MODELS}/eval_${RUN_NAME}_%j.out

source ${ENV}/bin/activate
cd ${EVAL_DIR}

bash scripts/zeroshot_deberta.sh \
    --model_path ${CKPT_ABS} \
    --langs "${LANGS}" \
    --batch_size 8 \
    --max_length 256
EOF

    JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
    echo "Submitted eval_${RUN_NAME}: job ${JOB_ID}  (langs: ${LANGS})"
    rm "$JOB_SCRIPT"
done
