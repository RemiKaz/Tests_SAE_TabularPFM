# Shared defaults of the campaign scripts (all_runs.sh, all_runs_parallel.sh, job_array.slurm).
# Source it, don't execute it. Each var keeps its value if the caller already set one, so
# `MODELS_ALL="mitra_v2" bash scripts/all_runs_parallel.sh` overrides it everywhere downstream.

DATASETS_ALL=${DATASETS_ALL:-"breast_cancer iris wine digits"}      # datasets/dataset_importer.py
MODELS_ALL=${MODELS_ALL:-"tabpfn_v3 tabpfn_v2 tabicl_v2 tabicl_v1 tabfm_v1 mitra_v2 mitra_v1"}
# The metrics computed for every run (utils/metrics.py). They all come from the same
# embeddings, so a layer is one sweep whatever the number of metrics; a rerun only asks for
# the metrics whose result file is still missing. To compare layers of one model on one
# dataset, use ccos and nmi: in synthetic checks they kept the right order between layers of
# different width and shape every time, while plain cosine inverts it when a layer has a
# large shared mean vector and CKA (any variant) when layers differ in spectrum. dcka is
# kept for the geometry view along one curve, where every metric orders levels correctly.
METRICS_ALL=${METRICS_ALL:-"cosine ccos nmi dcka"}

# The layers of a model. The models made of column, row and ICL stages are read after each
# stage; the others are one stack of identical layers, read early, in the middle and at the
# end. For TabICL and TabFM `last` is the end of the ICL stage, so it is not listed twice.
layers_of() {
    case "$1" in
        tabicl_v1 | tabicl_v2)
            echo "tf_col_early tf_col_middle tf_col tf_row_early tf_row_middle tf_row tf_icl"
            ;;
        tabfm_v1) echo "tf_col_1 tf_row_1 tf_col_2 tf_row_2 tf_icl" ;;
        tabpfn_v3) echo "tf_col tf_row tf_icl last" ;;
        tabpfn_v2 | mitra_v1 | mitra_v2) echo "early middle last" ;;
        *)
            echo "No layers defined for the model $1, add it to layers_of in scripts/config.sh" >&2
            return 1
            ;;
    esac
}

# Every (model, dataset) pair, in a fixed order, one per line as "model dataset" - the unit
# a Slurm array task picks one of, by its index.
all_pairs() {
    for model in $MODELS_ALL; do
        for dataset in $DATASETS_ALL; do
            echo "$model $dataset"
        done
    done
}

# The result folder name of one run, as run_exps.py writes it: "cosine" is left unsuffixed,
# so every result computed before --metric existed keeps its name.
run_name() {
    local model="$1" dataset="$2" exp="$3" layer="$4" metric="$5"
    local suffix=""
    [ "$metric" != "cosine" ] && suffix="_metric-$metric"
    echo "${model}_${dataset}_${exp}_layer-${layer}${suffix}"
}

# The metrics of METRICS_ALL still missing a result for one (model, dataset, layer), under
# $EXP (defaults to "all", as all_runs.sh does). Run from the project root.
missing_metrics() {
    local model="$1" dataset="$2" layer="$3" exp="${EXP:-all}" metric run missing=""
    for metric in $METRICS_ALL; do
        run=$(run_name "$model" "$dataset" "$exp" "$layer" "$metric")
        [ -f "results/$run/$run.json" ] || missing="$missing $metric"
    done
    echo $missing
}

# Whether every layer of a (model, dataset) pair already has a result for every metric.
pair_is_done() {
    local model="$1" dataset="$2" layer layers
    layers=$(layers_of "$model") || return 1
    for layer in $layers; do
        [ -z "$(missing_metrics "$model" "$dataset" "$layer")" ] || return 1
    done
}
