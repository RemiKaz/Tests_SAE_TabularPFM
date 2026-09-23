#!/bin/bash
# Run every experiment on every dataset, model and layer, spread over several GPUs at once,
# as ONE Slurm job array (scripts/job_array.slurm) instead of one job per (model, dataset):
# an array counts as a single submission, so it doesn't hit QOSMaxSubmitJobPerUserLimit the
# way looping `sbatch` does, and its own --array=...%THROTTLE caps how many tasks run at
# once, which respects whatever running-job/GPU limit the account has.
#
#     bash scripts/all_runs_parallel.sh                 # submit everything
#     DRY=1 bash scripts/all_runs_parallel.sh            # only print the sbatch command
#     THROTTLE=4 bash scripts/all_runs_parallel.sh       # at most 4 tasks running at once
#     METRICS_ALL="mi cka" bash scripts/all_runs_parallel.sh       # only these metrics
#     MODELS_ALL="mitra_v2" DATASETS_ALL="iris wine" bash scripts/all_runs_parallel.sh
#
# Extra arguments are passed to sbatch, e.g. to override job_array.slurm's #SBATCH settings:
#     bash scripts/all_runs_parallel.sh --qos=qos_gpu_a100-dev --time=02:00:00
#
# Run it from a login shell (not from inside an srun/sbatch allocation), after
# `python down_weight.py` (the compute nodes have no internet access). Results and
# skip-if-done work exactly as for job.slurm: rerunning this script skips what is done.

cd "$(dirname "$0")/.." || exit 1            # the project root, where job_array.slurm is
source scripts/config.sh

THROTTLE=${THROTTLE:-6}    # tasks running at once; raise/lower to the account's own limit

# Pairs whose every layer is already on disk (FORCE=1 redoes them, as all_runs.sh's own
# FORCE does per layer) don't get an array index, so a finished pair costs nothing to
# resubmit, not even a queued job slot.
indices=()
n_done=0
i=0
while read -r model dataset; do
    if [ -z "$FORCE" ] && pair_is_done "$model" "$dataset"; then
        n_done=$((n_done + 1))
    else
        indices+=("$i")
    fi
    i=$((i + 1))
done < <(all_pairs)

if [ ${#indices[@]} -eq 0 ]; then
    echo "All $n_done (model, dataset) pairs already done, nothing to submit"
    exit 0
fi
array_spec=$(IFS=,; echo "${indices[*]}")

if [ -n "$DRY" ]; then
    echo "MODELS_ALL=\"$MODELS_ALL\" DATASETS_ALL=\"$DATASETS_ALL\" \\"
    echo "  sbatch --array=$array_spec%$THROTTLE $* scripts/job_array.slurm"
    echo "${#indices[@]} pairs to run ($n_done already done, skipped), up to $THROTTLE at once"
elif sbatch --array="$array_spec%$THROTTLE" "$@" scripts/job_array.slurm; then
    echo "${#indices[@]} pairs to run ($n_done already done, skipped), up to $THROTTLE at once"
else
    echo "sbatch refused the submission, see the error above - nothing was queued" >&2
    exit 1
fi
