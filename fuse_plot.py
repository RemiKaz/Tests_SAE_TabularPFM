"""Fuse the results of a model on a dataset into a single figure, one curve per layer.

Every run of `run_exps.py` writes one json file per (model, dataset, experiment, layer). This
script collects the ones of a model and a dataset, and draws for every experiment a panel
with one curve per layer, so the layers can be compared on the same axes.

Examples:
    python fuse_plot.py --model tabicl_v2 --dataset breast_cancer
    python fuse_plot.py --model mitra_v2 --dataset breast_cancer --layers early middle last
    python fuse_plot.py --all                    # every (model, dataset, exp) found
    python fuse_plot.py --all --exp all           # only the "all" experiment group

The figure and a csv with the plotted numbers are written to `<results_dir>/_fused/`.
"""

import argparse
import csv
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb

from utils.metrics import METRIC_LABELS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BACKGROUND = "#fcfcfb"

# The layers are ordered by depth, so they wear one hue that gets darker with depth. It
# starts at step 250 (2.06:1 on the surface), the lightest an ordered mark may be. Up to 5
# layers the steps picked from this ramp clear the validator's ordinal checks (monotone
# lightness, adjacent gaps >= 0.06 in OKLCH L); a single hue cannot go further than that and
# stay inside the ordinal contrast floor, so past 5 layers, color alone cannot carry the
# distinction any more - MARKERS is the redundant channel that keeps working past that
# ceiling (shape, not lightness), paired with color one-to-one by depth index.
BLUE_RAMP = (
    "#86b6ef",  # 250
    "#6da7ec",  # 300
    "#5598e7",  # 350
    "#3987e5",  # 400
    "#2a78d6",  # 450
    "#256abf",  # 500
    "#1c5cab",  # 550
    "#184f95",  # 600
    "#104281",  # 650
    "#0d366b",  # 700
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "8")
MAX_DISCRETE_LAYERS = len(BLUE_RAMP)

# Named layers, in the order the data flows through the models that have them. Depths named
# by a fraction of the stack come first, then the layer indices, the stages and "last".
NAMED_ORDER = (
    "early",
    "middle",
    "tf_col_early",
    "tf_col_middle",
    "tf_col",
    "tf_col_1",
    "tf_row_1",
    "tf_col_2",
    "tf_row_2",
    "tf_row_early",
    "tf_row_middle",
    "tf_row",
    "tf_icl",
    "last",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default=None, help="Model of the runs, e.g. tabicl_v2")
    parser.add_argument("--dataset", default=None, help="Dataset of the runs, e.g. breast_cancer")
    parser.add_argument(
        "--exp",
        default=None,
        help='Experiment (or group) of the runs. Default: "perturbations" for a single plot, '
        "every experiment found for --all.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help='Metric of the runs (cosine, mi, cka - see utils/metrics.py). Default: "cosine" '
        "for a single plot, every metric found for --all. Runs of different metrics are never "
        "mixed into the same figure, their scales are not comparable.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fuse every (model, dataset, exp) combination found in --results_dir instead of "
        "one plot for --model/--dataset. Incompatible with --layers and --out.",
    )
    parser.add_argument(
        "--results_dir", default=DEFAULT_RESULTS_DIR, help="Directory of the run_exps.py results"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Layers to draw, in this order. Default: every layer found, by depth. Set it when\n"
        "mixing depth names (early, middle) with layer indices, whose depths cannot be compared.",
    )
    parser.add_argument(
        "--full_scale",
        action="store_true",
        help="Draw the y axis from 0 to 1 instead of zooming on the range of the curves",
    )
    parser.add_argument("--out", default=None, help="Path of the figure (png), single-plot mode only")
    args = parser.parse_args()

    if args.all:
        if args.layers is not None or args.out is not None:
            parser.error("--all cannot be combined with --layers or --out")
    elif args.model is None or args.dataset is None:
        parser.error("--model and --dataset are required (or pass --all)")
    return args


def layer_key(layer):
    """Sort key putting the layers in the order of the network."""
    if layer.lstrip("-").isdigit():
        return (1, int(layer), "")
    if layer in NAMED_ORDER:
        # early / middle name a depth, the others are stages, the indices sit in between
        rank = NAMED_ORDER.index(layer)
        return (0 if rank < 2 else 2, rank, "")
    return (3, 0, layer)  # a dotted submodule name


def load_runs(results_dir, model, dataset, exp, metric):
    """The results of every layer of a (model, dataset, experiment, metric), latest wins.

    Runs of different metrics are never merged into one dict: their values are not on
    comparable scales (mutual information is unbounded, cosine/CKA stay near [0, 1]).

    Returns:
        dict: layer -> list of experiment results, as written by `run_exps.py`.
    """
    runs = {}
    found = set()
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*.json"))):
        try:
            with open(path) as file:
                payload = json.load(file)
            args = payload["args"]
            # older result files predate --metric and were all cosine
            key = (args["model"], args["dataset"], args["exp"], args.get("metric", "cosine"))
            layer = str(args["layer"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        found.add(key)
        if key != (model, dataset, exp, metric):
            continue
        mtime = os.path.getmtime(path)
        if layer in runs and runs[layer][0] > mtime:
            continue
        if layer in runs:
            print(f"Several runs for layer {layer}, keeping the latest: {path}")
        runs[layer] = (mtime, payload["results"])

    if not runs:
        available = "\n".join(
            f"  --model {m} --dataset {d} --exp {e} --metric {me}" for m, d, e, me in sorted(found)
        )
        raise SystemExit(
            f"No result for model={model} dataset={dataset} exp={exp} metric={metric} in "
            f"{results_dir}.\nAvailable:\n{available or '  (nothing)'}"
        )
    return {layer: results for layer, (_, results) in runs.items()}


def find_combinations(results_dir, exp=None, metric=None):
    """Every (model, dataset, exp, metric) with at least one result on disk, in a stable order.

    Args:
        results_dir (str): The directory of the run_exps.py results.
        exp (str, optional): Keep only this experiment (or group). Defaults to None, every
            one found.
        metric (str, optional): Keep only this metric. Defaults to None, every one found.

    Returns:
        list: The (model, dataset, exp, metric) tuples, sorted.
    """
    combos = set()
    for path in glob.glob(os.path.join(results_dir, "*", "*.json")):
        try:
            with open(path) as file:
                run_args = json.load(file)["args"]
            combos.add(
                (run_args["model"], run_args["dataset"], run_args["exp"], run_args.get("metric", "cosine"))
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if exp is not None:
        combos = {combo for combo in combos if combo[2] == exp}
    if metric is not None:
        combos = {combo for combo in combos if combo[3] == metric}
    return sorted(combos)


def fuse_one(results_dir, model, dataset, exp, metric, full_scale, out_path=None):
    """Load, plot and save the fused figure of one combination, skipped-on-error.

    Returns:
        list or None: The layers plotted, or None if this combination had no usable result.
    """
    try:
        runs = load_runs(results_dir, model, dataset, exp, metric)
    except SystemExit as error:
        print(f"  {model} {dataset} {exp} {metric}: skipped, {error}")
        return None

    layers = sorted(runs, key=layer_key)
    # "cosine" left out of the file name, to match the file names run_exps.py itself writes.
    metric_suffix = "" if metric == "cosine" else f"_metric-{metric}"
    out_path = out_path or os.path.join(
        results_dir, "_fused", f"{model}_{dataset}_{exp}{metric_suffix}_layers.png"
    )
    title = f"{model} / {dataset} / {exp}: {METRIC_LABELS.get(metric, metric)} by layer"
    plot_layers(runs, layers, layer_colors(len(layers)), title, out_path, metric, full_scale)
    write_table(runs, layers, os.path.splitext(out_path)[0] + ".csv")
    return layers


def layer_colors(n_layers):
    """One color per layer, darker with depth.

    Evenly-spaced discrete steps of BLUE_RAMP up to its own length (10): past 5 the result
    no longer clears the ordinal contrast floor on its own (a single hue only has so much
    usable lightness range), which is why layers also get a distinct marker shape - see
    MARKERS. Beyond BLUE_RAMP's own length, falls back to interpolating between its stops,
    which only trades one real limitation (too few distinct hues) for a worse one (some
    layers sharing a color); not expected to trigger for any model in this project.
    """
    if n_layers == 1:
        return [BLUE_RAMP[4]]
    if n_layers <= len(BLUE_RAMP):
        return [BLUE_RAMP[round(i * (len(BLUE_RAMP) - 1) / (n_layers - 1))] for i in range(n_layers)]
    stops = [to_rgb(color) for color in BLUE_RAMP]
    colors = []
    for i in range(n_layers):
        position = i * (len(stops) - 1) / (n_layers - 1)
        low = int(position)
        high = min(low + 1, len(stops) - 1)
        fraction = position - low
        colors.append(tuple(a + (b - a) * fraction for a, b in zip(stops[low], stops[high])))
    return colors


def layer_markers(n_layers):
    """One marker shape per layer, paired with layer_colors by depth index.

    Cycles past MARKERS' own length rather than erroring: at that point some layers already
    share a marker (as layer_colors' own fallback lets them share a color), a plot that
    dense is past what any single encoding carries cleanly.
    """
    return [MARKERS[i % len(MARKERS)] for i in range(n_layers)]


def plot_layers(runs, layers, colors, title, out_path, metric="cosine", full_scale=False):
    # experiment name -> (title, xlabel), in the order they were run
    experiments = {}
    for layer in layers:
        for result in runs[layer]:
            experiments.setdefault(result["name"], (result["title"], result["xlabel"]))

    n_panels = len(experiments)
    n_cols = min(3, n_panels)
    n_rows = -(-n_panels // n_cols)
    # A legend past a handful of entries wraps into extra rows (below), needing a little
    # more headroom than a single row - a fixed extra inch of slack per extra row so it
    # isn't cramped; the exact top margin is measured after the legend is drawn, below,
    # rather than guessed (its real height depends on rendered font metrics, not a fixed
    # fraction of the figure).
    with_markers = len(layers) <= MAX_DISCRETE_LAYERS
    legend_cols = min(len(layers), 4) if with_markers else 0
    legend_rows = -(-len(layers) // legend_cols) if legend_cols else 0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows + 0.8 + 0.3 * max(0, legend_rows - 1)),
        facecolor=BACKGROUND,
        squeeze=False,
        sharey=True,
    )

    markers = layer_markers(len(layers)) if with_markers else [None] * len(layers)
    lowest, highest = math.inf, -math.inf
    for ax, (name, (panel_title, xlabel)) in zip(axes.flat, experiments.items()):
        ax.set_facecolor(BACKGROUND)
        for layer, color, marker in zip(layers, colors, markers):
            result = next((r for r in runs[layer] if r["name"] == name), None)
            if result is None:
                continue
            lowest = min(lowest, min(result["similarities"]))
            highest = max(highest, max(result["similarities"]))
            ax.plot(
                result["levels"],
                result["similarities"],
                color=color,
                linewidth=1.6,
                marker=marker,
                markersize=5.5 if marker in ("*", "P", "X") else 4.5,  # visually smaller shapes
                markeredgecolor=BACKGROUND,  # a gap of surface where the marks overlap
                markeredgewidth=1.0,
            )
        ax.set_title(panel_title, color=INK, fontsize=11)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_COLOR)
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABELS.get(metric, "Similarity to baseline"), color=MUTED, fontsize=9)
    if metric == "mi":
        # Unbounded (nats), and unlike cosine/cka does not start at a fixed value (even at
        # level 0 the perturbed embeddings only equal the baseline up to floating-point
        # noise, so per-dimension MI(baseline, baseline) is high but not exactly one fixed
        # number): zoom on the actual range seen, with a little padding either side.
        if full_scale:
            axes.flat[0].set_ylim(0, highest * 1.05)
        else:
            pad = (highest - lowest) * 0.05 or 0.05
            axes.flat[0].set_ylim(lowest - pad, highest + pad)
    else:
        # The curves start at 1 and the layers differ by a few percent: zoom on the range
        # they cover, on a round bound, unless the full scale is asked for.
        if full_scale:
            axes.flat[0].set_ylim(min(0.0, math.floor((lowest - 0.02) * 20) / 20), 1.05)
        else:
            axes.flat[0].set_ylim(min(0.95, math.floor((lowest - 0.02) * 20) / 20), 1.02)

    suptitle = fig.suptitle(title, color=INK, fontsize=13, y=0.985)

    if with_markers:
        # a legend, in the order of the depth, colors and shapes matching the lines. Anchored
        # just under the title and left to grow downward as rows are added.
        handles = [
            plt.Line2D([], [], color=color, linewidth=1.6, marker=marker, markersize=5.5)
            for color, marker in zip(colors, markers)
        ]
        legend = fig.legend(
            handles,
            layers,
            title="layer",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=legend_cols,
            frameon=False,
            labelcolor=SECONDARY,
            fontsize=9,
            title_fontsize=9,
        )
        legend.get_title().set_color(MUTED)
        top_element = legend
    else:
        # too many layers for color + shape alone: a depth scale, one cell per layer
        cax = fig.add_axes((0.2, 0.915, 0.6, 0.016))
        scale = ScalarMappable(norm=BoundaryNorm(range(len(layers) + 1), len(layers)), cmap=ListedColormap(colors))
        bar = fig.colorbar(scale, cax=cax, orientation="horizontal", ticks=[i + 0.5 for i in range(len(layers))])
        bar.ax.set_xticklabels(layers)
        bar.ax.tick_params(colors=SECONDARY, labelsize=8, length=0)
        bar.outline.set_visible(False)
        bar.ax.set_title("layer (shallow to deep)", color=MUTED, fontsize=9, pad=4)
        top_element = bar.ax

    # tight_layout first, for good inter-panel spacing, with the full figure still available;
    # only then is the top boundary pulled down to clear the legend/colorbar (whose real
    # height depends on rendered font metrics, so it is measured, not guessed). A plain
    # tight_layout(rect=...) call does not work for this: it still leaves its own heuristic
    # padding inside the given rect, which does not shrink to a requested value reliably -
    # subplots_adjust(top=...) is the one call that takes the figure-fraction value literally.
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    top_bottom = top_element.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y0
    fig.subplots_adjust(top=top_bottom - 0.03)
    suptitle.set_y(0.985)  # subplots_adjust can nudge the suptitle along with the rest back down

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_table(runs, layers, out_path):
    """The plotted numbers, one row per point, next to the figure."""
    with open(out_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment", "layer", "level", "value"])
        for layer in layers:
            for result in runs[layer]:
                for level, similarity in zip(result["levels"], result["similarities"]):
                    writer.writerow([result["name"], layer, level, similarity])


def main():
    args = parse_args()

    if args.all:
        combos = find_combinations(args.results_dir, args.exp, args.metric)
        if not combos:
            filters = [f" exp={args.exp}" if args.exp else "", f" metric={args.metric}" if args.metric else ""]
            raise SystemExit(f"No result found in {args.results_dir} for{''.join(filters) or ' any'} combination")
        print(f"{len(combos)} (model, dataset, exp, metric) combination(s) found")
        n_done = 0
        for model, dataset, exp, metric in combos:
            layers = fuse_one(args.results_dir, model, dataset, exp, metric, args.full_scale)
            if layers is not None:
                print(f"  {model} {dataset} {exp} {metric}: {len(layers)} layer(s)")
                n_done += 1
        print(f"Saved {n_done}/{len(combos)} plots to {os.path.join(args.results_dir, '_fused')}")
        return

    exp = args.exp or "perturbations"
    metric = args.metric or "cosine"
    runs = load_runs(args.results_dir, args.model, args.dataset, exp, metric)
    layers = args.layers if args.layers is not None else sorted(runs, key=layer_key)
    missing = [layer for layer in layers if layer not in runs]
    if missing:
        raise SystemExit(f"No result for layers {missing}, found: {sorted(runs, key=layer_key)}")

    metric_suffix = "" if metric == "cosine" else f"_metric-{metric}"
    out_path = args.out or os.path.join(
        args.results_dir, "_fused", f"{args.model}_{args.dataset}_{exp}{metric_suffix}_layers.png"
    )
    title = f"{args.model} / {args.dataset} / {exp}: {METRIC_LABELS.get(metric, metric)} by layer"
    plot_layers(runs, layers, layer_colors(len(layers)), title, out_path, metric, args.full_scale)
    write_table(runs, layers, os.path.splitext(out_path)[0] + ".csv")
    print(f"Layers {layers}\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
