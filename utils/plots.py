import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.metrics import METRIC_LABELS

INK = "#0b0b0b"
MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SERIES_COLOR = "#2a78d6"
BACKGROUND = "#fcfcfb"


def plot_experiments(results, out_dir, file_name="experiments.png", suptitle=""):
    """Plot one panel of similarity to baseline vs. perturbation intensity per experiment.

    Args:
        results (list): A list of result dicts, as returned by `run_experiment`.
        out_dir (str): The output directory.
        file_name (str, optional): The name of the figure. Defaults to "experiments.png".
        suptitle (str, optional): The title of the whole figure. Defaults to "".

    Returns:
        str: The path of the written figure.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, file_name)

    n_panels = len(results)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), facecolor=BACKGROUND, squeeze=False
    )

    for ax, result in zip(axes.flat, results):
        _plot_panel(ax, result)
    for ax in axes.flat[n_panels:]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved plot to {path}")
    return path


def _plot_panel(ax, result):
    ax.set_facecolor(BACKGROUND)
    ax.plot(
        result["levels"],
        result["similarities"],
        color=SERIES_COLOR,
        linewidth=2,
        marker="o",
        markersize=6,
    )
    ax.set_title(result["title"], color=INK, fontsize=11)
    ax.set_xlabel(result["xlabel"], color=MUTED, fontsize=9)
    metric = result.get("metric", "cosine")  # older result files predate the "metric" key
    ax.set_ylabel(METRIC_LABELS.get(metric, "Similarity to baseline"), color=MUTED, fontsize=9)
    if metric == "mi":
        ax.set_ylim(bottom=0)  # mutual information (nats) has no fixed upper bound
    else:
        # the others are about 1 when unchanged; dcka can dip slightly below 0 (estimation
        # noise around 0), cosine down to -1, so the bottom follows the data when it does
        ax.set_ylim(min(0, min(result["similarities"]) - 0.05), 1.05)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
