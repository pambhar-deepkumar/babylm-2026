#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --time 5:00:00
#SBATCH --output /dss/dsshome1/07/go82qok2/git/nour/scripts/%j_train_tokenizers_zho.out

source ~/babylm_env/bin/activate
cd ~/git/nour

for VOCAB in 4000 16000 32000; do
  echo "=== Training Chinese tokenizer, vocab_size=${VOCAB} ==="
  python scripts/train_tokenizer.py \
    --input  babylm_2026/data/zho/texts1.txt \
    --output babylm_2026/tokenizers \
    --lang   zho \
    --vocab_size ${VOCAB} \
    --name   tokenizer
done

echo "=== Done: Chinese tokenizers ==="
