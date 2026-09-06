#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 8:00:00
#SBATCH --cpus-per-task 8
#SBATCH --output /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/models/nld_8k_isotoken_%j.out

source /dss/dsshome1/07/go82qok2/babylm_env/bin/activate
cd ~/git/nour

# Iso-token-budget rerun of nld_8k: capped at --max_steps 67275 to match
# nld_baseline (40k vocab)'s total training steps, so both vocab-size
# conditions see the same number of training tokens/gradient updates.
# Fixes the confound where "10 epochs" gave 8k vocab ~18% more tokens
# than 40k vocab due to higher subword fertility.
python LukasLM/train_mask.py   --train_data babylm_2026/data/nld/texts2.txt  --valid_data babylm_2026/data/nld/texts1.txt   --tokenizer babylm_2026/tokenizers/nld_spm_8k.model   --output_path babylm_2026/models/nld_8k_isotoken   --hidden_size 768   --intermediate_size 3072   --batch_size 256   --lr 0.001   --epochs 10   --max_steps 67275   --max_seq_len "0:64,5:256"   --mask_decay 0.1   --all_checkpoints   --lamb   --cpus 8
