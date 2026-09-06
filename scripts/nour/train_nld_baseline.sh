#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 8:00:00
#SBATCH --cpus-per-task 8
#SBATCH --output /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/models/nld_baseline_%j.out

source /dss/dsshome1/07/go82qok2/babylm_env/bin/activate
cd ~/git/nour

python LukasLM/train_mask.py   --train_data babylm_2026/data/nld/texts2.txt  --valid_data babylm_2026/data/nld/texts1.txt   --tokenizer babylm_2026/tokenizers/nld_spm.model   --output_path babylm_2026/models/nld_baseline   --hidden_size 768   --intermediate_size 3072   --batch_size 256   --lr 0.001   --epochs 10   --max_seq_len "0:64,5:256"   --mask_decay 0.1   --all_checkpoints   --lamb   --cpus 8
