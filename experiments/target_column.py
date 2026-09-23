import numpy as np

from datasets.tabular_dataset import TabularDataset
from experiments.base_experiment import Experiment


class AltTarget(Experiment):
    """Predict another column of X instead of y.

    The alternative target is binarized with a median split and dropped from the
    features, so the intensity axis is the index of the column used as target.
    """

    name = "alt_target"
    title = "Predict a different column of X instead of y"
    xlabel = "Column index used as alternative target"

    def levels(self, dataset, n_levels):
        # One level per column: n_levels is ignored here.
        return np.arange(dataset.X.shape[1])

    def perturb(self, dataset, level, rng):
        y_alt = (dataset.X[:, level] > np.median(dataset.X[:, level])).astype(int)
        X_alt = np.delete(dataset.X, level, axis=1)
        return TabularDataset.from_xy(
            dataset.name, X_alt, y_alt, test_size=dataset.test_size, seed=dataset.seed
        )
