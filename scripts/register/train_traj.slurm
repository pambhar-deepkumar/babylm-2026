#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 04:00:00
#SBATCH --cpus-per-task 8
#SBATCH --output %j_%x.out

# Trajectory training for the AoA / acquisition-curve study. Saves FULL
# checkpoints (optimizer state included) at the token milestones defined in
# train_gpt2.py, so an A/B/C/D acquisition curve sits on an identical x-axis.
# Same recipe as train_arm_b.slurm; only corpus/output/seed differ.
#
# NB space: ~22 full checkpoints x ~1.2 GB = ~26 GB per arm. Run arms SERIALLY
# and back up + prune each arm's intermediates before starting the next.
#
#   sbatch --job-name traj_b --export=ALL,CORPUS=data/bb26_simplified.train,OUT=output/traj_b,SEED=42 scripts/register/train_traj.slurm
set -euo pipefail
source ~/gpt2_env/bin/activate   # example env; point at your own venv
cd "${SLURM_SUBMIT_DIR:-.}"       # run from the repo root you sbatch from
export HF_HUB_OFFLINE=1
echo "CORPUS=$CORPUS OUT=$OUT SEED=$SEED (trajectory / milestones on)"
python scripts/register/train_gpt2.py \
  --train_file "$CORPUS" --model_path baseline_gpt2_2026 \
  --output_dir "$OUT" --epochs 20 --lr 5e-4 --per_device_batch 32 \
  --grad_accum 1 --seq_len 1024 --seed "$SEED" --bf16 --save_milestones
