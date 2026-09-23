"""Run invariance experiments on a tabular foundation model.

Example:
    python run_exps.py --model tabpfn_v3 --dataset breast_cancer --exp perturbations --layer last
"""

import argparse
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from datasets import SKLEARN_LOADERS, dataset_importer
from experiments import EXPERIMENT_GROUPS, EXPERIMENTS, experiment_importer, run_experiment
from models import MODEL_NAMES, model_importer
from utils import METRICS, plot_experiments, save_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, default="tabpfn_v3", choices=list(MODEL_NAMES), help="Model to probe"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=list(SKLEARN_LOADERS),
        help="Dataset to run the experiments on",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="perturbations",
        choices=list(EXPERIMENTS) + list(EXPERIMENT_GROUPS),
        help="Experiment, or group of experiments, to run",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="last",
        help="Layer position the embeddings are read at: 'last' for the native "
        "embeddings, an index in the layer stack, or a dotted submodule name",
    )
    parser.add_argument(
        "--list_layers",
        action="store_true",
        help="Print the layers the embeddings can be read at, then exit",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="+",
        default=["cosine"],
        choices=list(METRICS),
        help="How the perturbed embeddings are compared to the baseline: 'cosine' (per row, "
        "then averaged), 'mi' (mutual information, per dimension, then averaged) or 'cka' "
        "(Centered Kernel Alignment, jointly over all dimensions), see utils/metrics.py. "
        "Several can be given, they are all computed from the same embeddings (one sweep), "
        "and each is saved to its own result folder.",
    )

    parser.add_argument("--model_path", type=str, default="default", help="Checkpoint path")
    parser.add_argument("--n_estimators", type=int, default=1, help="Size of the model ensemble")
    parser.add_argument(
        "--feature_slot",
        type=str,
        default="target",
        choices=["target", "mean", "flatten"],
        help="How per-cell activations are reduced to one vector per row",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    parser.add_argument("--test_size", type=float, default=0.33, help="Test set proportion")
    parser.add_argument("--n_levels", type=int, default=20, help="Number of intensity steps")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the perturbations")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed of the train/test split")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--eval", action="store_true", help="Also report accuracy and ROC AUC on the clean data"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model = model_importer(
        args.model,
        device=args.device,
        layer=args.layer,
        model_path=args.model_path,
        n_estimators=args.n_estimators,
        feature_slot=args.feature_slot,
        seed=args.seed,
    )
    dataset = dataset_importer(args.dataset, test_size=args.test_size, seed=args.split_seed)

    if args.list_layers:
        model.fit(dataset.X_train, dataset.y_train)
        layers = model.list_layers()
        print("\nLayer stack (addressable by index):")
        if not layers["layer_stack"]:
            print("  none found for this model, address a submodule by name")
        for index, name in layers["layer_stack"]:
            print(f"  {index:3d}  {name}")
        for depth, name in layers["depths"].items():
            print(f"  {depth:>6s}  = {name}")
        print("\nAll submodules (addressable by name):")
        for name in layers["all_modules"]:
            print(f"       {name}")
        return

    experiments = experiment_importer(args.exp)
    rng = np.random.default_rng(args.seed)
    metrics = list(dict.fromkeys(args.metric))  # drop duplicates, keep order
    by_experiment = [
        run_experiment(experiment, model, dataset, rng=rng, n_levels=args.n_levels, metrics=metrics)
        for experiment in experiments
    ]

    for metric in metrics:
        results = [per_metric[metric] for per_metric in by_experiment]
        # "cosine" is left out of the name: every existing result predates --metric and was
        # cosine, so this keeps their names (and the skip-if-done checks in scripts/) unchanged.
        metric_suffix = "" if metric == "cosine" else f"_metric-{metric}"
        run_name = f"{args.model}_{args.dataset}_{args.exp}_layer-{args.layer}{metric_suffix}"
        out_dir = os.path.join(args.out_dir, run_name)
        # one metric per file, so readers like fuse_plot.py see a single "metric" in args
        save_results(
            {"args": {**vars(args), "metric": metric}, "results": results},
            out_dir,
            file_name=f"{run_name}.json",
        )
        metric_note = f" ({metric})" if metric != "cosine" else ""
        plot_experiments(
            results,
            out_dir,
            file_name=f"{run_name}.png",
            suptitle=f"{args.model} / {args.dataset} / layer {args.layer}{metric_note}",
        )

    if args.eval:
        model.fit(dataset.X_train, dataset.y_train)
        probabilities = model.predict_proba(dataset.X_test)
        if probabilities.shape[1] == 2:
            print("ROC AUC:", roc_auc_score(dataset.y_test, probabilities[:, 1]))
        else:
            print("ROC AUC:", roc_auc_score(dataset.y_test, probabilities, multi_class="ovr"))
        print("Accuracy:", accuracy_score(dataset.y_test, model.predict(dataset.X_test)))


if __name__ == "__main__":
    main()
