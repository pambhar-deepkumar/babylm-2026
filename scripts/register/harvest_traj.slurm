#!/bin/bash
#SBATCH --partition lrz-dgx-a100-80x8,lrz-hgx-a100-80x4,lrz-hgx-h100-94x4
#SBATCH --gres gpu:1
#SBATCH --time 00:30:00
#SBATCH --cpus-per-task 8
#SBATCH --output %j_%x.out

# Harvest per-word surprisal across one arm's 22 milestone checkpoints (AoA).
# Pass ARM_DIR / ARM_NAME / OUT via --export:
#   sbatch --job-name harv_a --export=ALL,ARM_DIR=output/traj_a,ARM_NAME=A,OUT=results/register/aoa/traj_a_surprisal.csv scripts/register/harvest_traj.slurm
set -euo pipefail
source ~/gpt2_env/bin/activate   # example env; point at your own venv
cd "${SLURM_SUBMIT_DIR:-.}"       # run from the repo root you sbatch from
export HF_HUB_OFFLINE=1
python scripts/register/harvest_surprisal.py --arm_dir "$ARM_DIR" --arm_name "$ARM_NAME" \
  --probes data/aoa_probes.json --out "$OUT"
echo "HARVEST DONE -> $OUT"
