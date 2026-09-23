#!/bin/bash
#SBATCH --job-name=sae_tabpfm
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --account=ycq@a100          # kept in sync with job.slurm by hand, same settings
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-t3       # MaxSubmitPU=10000; the dev QOS caps it at 10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
# Shorter than the 20h qos_gpu_a100-t3 allows, for faster backfill scheduling, see job.slurm.
#SBATCH --time=04:00:00

# One array task = one (model, dataset) pair, every layer of that model done in sequence.
# Submitted by scripts/all_runs_parallel.sh, which sets --array from the pair count and
# %throttle from THROTTLE - don't sbatch this file directly, its --array above is a
# placeholder overridden by that script's own --array flag.
#SBATCH --array=0

cd "$SLURM_SUBMIT_DIR"
source scripts/config.sh

# The Nth (model, dataset) pair, 0-indexed, matching the array task id.
pair=$(all_pairs | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$pair" ]; then
    echo "No pair at index $SLURM_ARRAY_TASK_ID (only $(all_pairs | wc -l) pairs)" >&2
    exit 1
fi
read -r MODELS DATASETS <<<"$pair"
export MODELS DATASETS
echo "Array task $SLURM_ARRAY_TASK_ID: model=$MODELS dataset=$DATASETS"

# The A100 nodes take the modules of their own architecture, chosen from the account.
case "$SLURM_JOB_ACCOUNT" in
    *@a100) export JZ_ARCH=a100 ;;
esac

source setup_env.sh
nvidia-smi --query-gpu=name --format=csv,noheader  # the GPU the results come from

# The compute nodes have no internet access, the checkpoints have to be in $HF_HOME
# already. Run `python down_weight.py` on a login node if they are not.
export HF_HUB_OFFLINE=1

bash scripts/all_runs.sh
