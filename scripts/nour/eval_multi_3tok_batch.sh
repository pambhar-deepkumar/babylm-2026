#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH -o /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/models/eval_multi_run3_fixed_%j.out

source /dss/dsshome1/07/go82qok2/babylm_env/bin/activate

CHECKPOINT=/dss/dsshome1/07/go82qok2/git/nour/babylm_2026/models/multi_run3_3tok/checkpoint-66065
TOKENIZERS=/dss/dsshome1/07/go82qok2/git/nour/babylm_2026/tokenizers
SCRIPT=/dss/dsshome1/07/go82qok2/git/nour/babylm-eval/multilingual/scripts/zeroshot_deberta.sh

echo "=== multi_run3_3tok: eng with eng_8k preseg ==="
bash $SCRIPT \
    --model_path $CHECKPOINT \
    --langs "eng" \
    --preseg_model $TOKENIZERS/eng_tokenizer_8k.model \
    --batch_size 8 --max_length 256

echo "=== multi_run3_3tok: nld with nld_8k preseg ==="
bash $SCRIPT \
    --model_path $CHECKPOINT \
    --langs "nld" \
    --preseg_model $TOKENIZERS/nld_spm_8k.model \
    --batch_size 8 --max_length 256

echo "=== multi_run3_3tok: zho with zho_8k preseg ==="
bash $SCRIPT \
    --model_path $CHECKPOINT \
    --langs "zho" \
    --preseg_model $TOKENIZERS/zho_spm_8k.model \
    --batch_size 8 --max_length 256

