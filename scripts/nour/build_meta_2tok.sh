#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --time 2:00:00
#SBATCH --output /dss/dsshome1/07/go82qok2/git/nour/scripts/%j_build_meta_run2.out

source ~/babylm_env/bin/activate
cd ~/git/nour

python scripts/build_meta_tokenizer.py \
  --lang eng --tokenizer /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/tokenizers/engnld_tokenizer_16k.model --train_file /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/data/budget100m/eng_train_33m.txt \
  --lang nld --tokenizer /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/tokenizers/engnld_tokenizer_16k.model --train_file /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/data/budget100m/nld_train_33m.txt \
  --lang zho --tokenizer /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/tokenizers/zho_spm_8k.model            --train_file /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/data/budget100m/zho_train_33m.txt \
  --valid_lang eng --valid_file /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/data/budget100m/eng_valid_10m.txt \
  --out_dir /dss/dsshome1/07/go82qok2/git/nour/babylm_2026/data/budget100m --out_prefix run2
