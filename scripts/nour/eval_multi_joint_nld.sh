#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH -o /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/models/eval_run1_nld_%j.out

source /dss/dsshome1/07/go82qok2/babylm_env/bin/activate

bash /dss/dsshome1/07/go82qok2/git/nour/babylm-eval/multilingual/scripts/zeroshot_deberta.sh \
    --model_path /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/models/multi_run1_joint/checkpoint-63255 \
    --langs "nld" \
    --batch_size 8 --max_length 256

