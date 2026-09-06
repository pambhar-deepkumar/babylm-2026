#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --time 2:00:00
#SBATCH --output /dss/dsshome1/07/go82qok2/git/nour/scripts/%j_train_combo_tokenizers.out

source ~/babylm_env/bin/activate
cd ~/git/nour
python scripts/train_combo_tokenizers.py
