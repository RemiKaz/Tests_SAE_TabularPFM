#!/bin/bash
#SBATCH --job-name=sae_tabpfm
#SBATCH --output=logs/%x_%j.out
# No --error: stderr (tracebacks, FAILED lines) goes to the --output file, in order.
#SBATCH --account=ycq@a100          # `idrproj` lists the accounts, suffix = GPU partition
#SBATCH -C a100                     # the A100 nodes
# 2h max, MaxSubmitPU=10: fine for a single job like this one (starts sooner than t3), but
# too low for the parallel array - scripts/all_runs_parallel.sh submits its own array job
# (scripts/job_array.slurm) on qos_gpu_a100-t3 instead (20h max, MaxSubmitPU=10000).
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8           # the share of cores of one GPU on the A100 nodes
#SBATCH --hint=nomultithread
# Shorter than the 20h qos_gpu_a100-t3 allows: a request this size is easier for the
# scheduler to slot into a gap between bigger jobs (backfill), so it tends to start sooner.
# A run that doesn't finish in time is killed; results are only written when a (model,
# dataset, layer) run completes, so resubmitting picks up wherever it stopped, at the cost
# of redoing whatever the killed run hadn't finished writing yet.
#SBATCH --time=00:58:00

# Runs every experiment on every dataset, model and layer, see scripts/all_runs.sh - a bare
# `sbatch job.slurm` does the whole campaign on one GPU, which the 2h dev cap above will
# not fit; for that, use `bash scripts/all_runs_parallel.sh` instead, which spreads it over
# several GPUs at once. This file is for a single run or a small subset that fits in 2h:
#     mkdir -p logs        # once, Slurm opens the output files itself
#     MODELS="mitra_v2" DATASETS="iris" sbatch job.slurm
# For a V100 node instead, override the five settings when submitting (10 cores per GPU):
#     sbatch --account=ycq@v100 -C v100 --qos=qos_gpu-t3 --cpus-per-task=10 job.slurm

cd "$SLURM_SUBMIT_DIR"

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
