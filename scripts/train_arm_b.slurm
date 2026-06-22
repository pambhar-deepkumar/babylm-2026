#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 04:00:00
#SBATCH --cpus-per-task 8
#SBATCH --output %j_arm_b.out

# Train GPT-2 on the register-simplified corpus (Arm B), matching the official
# 2026 Strict-Small baseline recipe. Run from the project root on LRZ:
#   sbatch scripts/train_arm_b.slurm
#
# One-time prep on the LOGIN node (compute nodes may have no internet) — fetch
# the baseline config + tokenizer once into a local dir:
#   source ~/babylm_env/bin/activate
#   huggingface-cli download BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict-Small \
#     config.json tokenizer.json tokenizer_config.json special_tokens_map.json \
#     generation_config.json --local-dir baseline_gpt2_2026

set -euo pipefail
source ~/gpt2_env/bin/activate   # dedicated env (NOT the shared ~/babylm_env)
cd ~/git/deep/babylm-2026
export HF_HUB_OFFLINE=1   # use the pre-downloaded baseline dir; no network on compute

python scripts/train_gpt2.py \
  --train_file data/bb26_simplified.train \
  --model_path baseline_gpt2_2026 \
  --output_dir output/arm_b_simplified \
  --epochs 20 \
  --lr 5e-4 \
  --per_device_batch 32 \
  --grad_accum 1 \
  --seq_len 1024 \
  --seed 42 \
  --bf16
