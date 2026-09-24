#!/bin/bash
# Run the campaign on qos_gpu_a100-dev (starts sooner than t3) instead of the array in
# scripts/all_runs_parallel.sh, while respecting dev's MaxSubmitPU=10: a hard cap on how
# many jobs you may have submitted at once under that QOS, running and pending together.
#
# Each call submits one batch - one job.slurm per pending (model, dataset) pair, up to
# however many of the 10 slots are free right now - then exits, it does not wait or loop.
# A pair whose layers don't all finish inside job.slurm's own --time is simply still
# "pending" afterwards, same as any other unfinished pair, and gets resubmitted next call
# (all_runs.sh's own skip-if-done means nothing already written is redone).
#
# Repeat until it reports nothing left:
#     bash scripts/all_runs_dev.sh                     # submit what fits
#     squeue -u $USER                                  # wait for those to clear
#     bash scripts/all_runs_dev.sh                      # submit the next batch, etc.
#     DRY=1 bash scripts/all_runs_dev.sh                # only print what would be submitted
#     METRICS_ALL="mi cka" bash scripts/all_runs_dev.sh       # only these metrics
#     MODELS_ALL="mitra_v2" DATASETS_ALL="iris wine" bash scripts/all_runs_dev.sh

MAX_SUBMIT=${MAX_SUBMIT:-10}   # qos_gpu_a100-dev's MaxSubmitPU - sacctmgr show qos to check
QOS=${QOS:-qos_gpu_a100-dev}

cd "$(dirname "$0")/.." || exit 1
source scripts/config.sh
mkdir -p logs

# Jobs already queued or running, by name (<model>_<dataset>, set at submission below):
# those pairs are not done on disk yet, but submitting them again would compute the same
# layers twice in parallel.
queued=" $(squeue -h -u "$USER" -o %j | tr '\n' ' ') "

pending=()
n_queued=0
while read -r model dataset; do
    if [ -z "$FORCE" ] && pair_is_done "$model" "$dataset"; then
        continue
    fi
    if [[ "$queued" == *" ${model}_${dataset} "* ]]; then
        n_queued=$((n_queued + 1))
        continue
    fi
    pending+=("$model $dataset")
done < <(all_pairs)

if [ ${#pending[@]} -eq 0 ]; then
    echo "Nothing to submit: $n_queued pair(s) already queued or running, the rest done"
    exit 0
fi

in_flight=$(squeue -h -u "$USER" --qos="$QOS" | wc -l)
slots=$((MAX_SUBMIT - in_flight))
if [ "$slots" -le 0 ]; then
    echo "Already $in_flight jobs on $QOS (cap $MAX_SUBMIT), wait for some to finish (squeue -u \$USER) and rerun"
    exit 0
fi

n=0
for pair in "${pending[@]}"; do
    [ "$n" -ge "$slots" ] && break
    read -r model dataset <<<"$pair"
    if [ -n "$DRY" ]; then
        echo "MODELS=$model DATASETS=$dataset sbatch --job-name=${model}_${dataset} job.slurm"
    else
        MODELS=$model DATASETS=$dataset sbatch --job-name="${model}_${dataset}" job.slurm
    fi
    n=$((n + 1))
done

echo "$n submitted, $((${#pending[@]} - n)) left to submit later, $n_queued pair(s) already queued or running (cap $MAX_SUBMIT jobs)"
if [ "$n" -lt "${#pending[@]}" ]; then
    echo "Rerun once these clear (squeue -u \$USER) to submit the rest"
fi
exit 0
