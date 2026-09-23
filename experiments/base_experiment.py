from abc import ABC, abstractmethod

import numpy as np

from utils.metrics import METRICS


class Experiment(ABC):
    """A sweep of perturbations of growing intensity applied to a dataset.

    An experiment only says which intensities are swept and how the dataset is
    perturbed at a given intensity: `run_experiment` does the fitting, the embedding
    and the comparison to the unperturbed baseline.

    Attributes:
        name (str): The short name of the experiment, used on the command line.
        title (str): The title of the plot panel.
        xlabel (str): The label of the x axis of the plot panel.
    """

    name = ""
    title = ""
    xlabel = ""

    @abstractmethod
    def levels(self, dataset, n_levels):
        """The perturbation intensities swept by the experiment.

        Args:
            dataset (TabularDataset): The unperturbed dataset.
            n_levels (int): The number of intensity steps asked for.

        Returns:
            np.ndarray: The intensities.
        """

    @abstractmethod
    def perturb(self, dataset, level, rng):
        """Apply the perturbation at one intensity.

        Args:
            dataset (TabularDataset): The unperturbed dataset.
            level (int): The intensity of the perturbation.
            rng (np.random.Generator): The random generator.

        Returns:
            TabularDataset: The perturbed dataset.
        """


def run_experiment(experiment, model, dataset, rng=None, n_levels=20, metrics=("cosine",)):
    """Run one experiment: similarity to the baseline vs. perturbation intensity.

    Fitting the model and reading its embeddings is what costs; comparing them to the
    baseline is cheap next to that. So every metric asked for is computed on the same
    embeddings at each level, rather than rerunning the whole sweep once per metric.

    Args:
        experiment (Experiment): The experiment to run.
        model (BaseTabularPFM): The model whose embeddings are compared.
        dataset (TabularDataset): The unperturbed dataset.
        rng (np.random.Generator, optional): The random generator. Defaults to None.
        n_levels (int, optional): The number of intensity steps. Defaults to 20.
        metrics (tuple, optional): Names of metrics in `utils.metrics.METRICS` used to
            compare the perturbed embeddings to the baseline. Defaults to ("cosine",).

    Returns:
        dict: metric name -> the experiment name, title, xlabel, metric, levels and values.
    """
    rng = np.random.default_rng() if rng is None else rng
    print(f"Running experiment {experiment.name} ...")

    model.fit(dataset.X_train, dataset.y_train)
    baseline = model.get_embeddings(dataset.X_test)

    levels = np.asarray(experiment.levels(dataset, n_levels))
    values = {metric: [] for metric in metrics}
    for level in levels:
        perturbed = experiment.perturb(dataset, int(level), rng)
        model.fit(perturbed.X_train, perturbed.y_train)
        embeds = model.get_embeddings(perturbed.X_test)
        for metric in metrics:
            values[metric].append(METRICS[metric](embeds, baseline))
        print(f"  level {level}: " + ", ".join(f"{m}={values[m][-1]:.4f}" for m in metrics))

    return {
        metric: {
            "name": experiment.name,
            "title": experiment.title,
            "xlabel": experiment.xlabel,
            "metric": metric,
            "levels": levels.tolist(),
            "similarities": values[metric],
        }
        for metric in metrics
    }
