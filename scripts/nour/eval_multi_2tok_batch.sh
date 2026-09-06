#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH -o /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/models/eval_multi_run2_fixed_%j.out

source /dss/dsshome1/07/go82qok2/babylm_env/bin/activate

CHECKPOINT=/dss/dsshome1/07/go82qok2/git/nour/babylm_2026/models/multi_run2_2tok/checkpoint-63320
TOKENIZERS=/dss/dsshome1/07/go82qok2/git/nour/babylm_2026/tokenizers
SCRIPT=/dss/dsshome1/07/go82qok2/git/nour/babylm-eval/multilingual/scripts/zeroshot_deberta.sh

echo "=== multi_run2_2tok: eng+nld with engnld_16k preseg ==="
bash $SCRIPT \
    --model_path $CHECKPOINT \
    --langs "eng nld" \
    --preseg_model $TOKENIZERS/engnld_tokenizer_16k.model \
    --batch_size 8 --max_length 256

echo "=== multi_run2_2tok: zho with zho_8k preseg ==="
bash $SCRIPT \
    --model_path $CHECKPOINT \
    --langs "zho" \
    --preseg_model $TOKENIZERS/zho_spm_8k.model \
    --batch_size 8 --max_length 256

