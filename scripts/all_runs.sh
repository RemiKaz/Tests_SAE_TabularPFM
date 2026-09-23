#!/bin/bash
# Run every experiment on every dataset, model and layer, one after the other. It is what
# job.slurm runs, and it can also be run by hand inside an allocation. MODELS / DATASETS
# restrict it, e.g. to the one (model, dataset) pair a job_array.slurm task picked.
#
# The runs whose results exist are skipped, so after a time-out `sbatch job.slurm` again
# carries on where it stopped. FORCE=1 redoes them.
#
# Results go to results/<model>_<dataset>_<exp>_layer-<layer>/, and the curves of all the
# layers of a model are put on one figure with:
#     python ../fuse_plot.py --model <model> --dataset <dataset> --exp all

cd "$(dirname "$0")/.." || exit 1            # the project root, where run_exps.py is
source scripts/config.sh

EXP=${EXP:-all}                              # "all" is every experiment, see experiments/
DATASETS=${DATASETS:-$DATASETS_ALL}
MODELS=${MODELS:-$MODELS_ALL}

failed=0
for model in $MODELS; do
    layers=$(layers_of "$model") || exit 1
    for dataset in $DATASETS; do
        for layer in $layers; do
            run="${model}_${dataset}_${EXP}_layer-${layer}"
            # Only the metrics without a result yet: one sweep computes all of them.
            metrics=$METRICS_ALL
            [ -z "$FORCE" ] && metrics=$(missing_metrics "$model" "$dataset" "$layer")
            if [ -z "$metrics" ]; then
                echo "Skipping $run, done";continue              
            fi
            echo "Running $run ($metrics)"
            # A failing run must not stop the others.
            if ! python -u run_exps.py \
                --model "$model" --dataset "$dataset" --exp "$EXP" --layer "$layer" \
                --metric $metrics; then
                echo "FAILED: $run" >&2
                failed=$((failed + 1))
            fi
        done
    done
done

echo "Done, $failed runs failed"
exit $((failed > 0))
