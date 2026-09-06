#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --time 1:00:00
#SBATCH --output %j_train_tokenizers.out

source ~/babylm_env/bin/activate
cd ~/git/nour

#echo '=== Training Dutch tokenizer ==='
#python scripts/train_tokenizer.py     --input  src/babylm_2026/data/nld/dutch_train.txt     --output src/babylm_2026/tokenizers     --lang   nld

echo '=== Training Chinese tokenizer ==='
python scripts/train_tokenizer.py     --input  src/babylm_2026/data/zho/chinese_train.txt     --output src/babylm_2026/tokenizers     --lang   zho

echo '=== Done ==='
