#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --time 5:00:00
#SBATCH --output /dss/dsshome1/07/go82qok2/git/nour/scripts/%j_train_tokenizers_eng.out

source ~/babylm_env/bin/activate
cd ~/git/nour

for VOCAB in 4000 8000 16000 32000 40000; do
  echo "=== Training English tokenizer, vocab_size=${VOCAB} ==="
  python scripts/train_tokenizer.py \
    --input  babylm_2026/data/eng/eng_train.txt \
    --output babylm_2026/tokenizers \
    --lang   eng \
    --vocab_size ${VOCAB} \
    --name   tokenizer
done

echo "=== Done: English tokenizers ==="
