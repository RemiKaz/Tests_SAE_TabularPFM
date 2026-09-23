import numpy as np

from experiments.base_experiment import Experiment
from utils.utils import permute_axis


class RowShuffle(Experiment):
    """Shuffle the train rows, jointly in X_train and y_train so the pairing is kept."""

    name = "row_shuffle"
    title = "Shuffle train rows"
    xlabel = "n_permut (row swaps)"

    def levels(self, dataset, n_levels):
        return np.linspace(0, dataset.n_train, n_levels, dtype=int)

    def perturb(self, dataset, level, rng):
        train = np.column_stack([dataset.X_train, dataset.y_train])
        train = permute_axis(train, axis=0, n_permut=level, rng=rng)
        return dataset.perturbed(
            X_train=train[:, :-1], y_train=train[:, -1].astype(dataset.y_train.dtype)
        )


class ColShuffle(Experiment):
    """Shuffle the columns of X, with the same column order on the train and test sets."""

    name = "col_shuffle"
    title = "Shuffle columns"
    xlabel = "n_permut (column swaps)"

    def levels(self, dataset, n_levels):
        return np.linspace(0, dataset.n_features, n_levels, dtype=int)

    def perturb(self, dataset, level, rng):
        X = permute_axis(
            np.vstack([dataset.X_train, dataset.X_test]), axis=1, n_permut=level, rng=rng
        )
        return dataset.perturbed(X_train=X[: dataset.n_train], X_test=X[dataset.n_train :])


class ColRemove(Experiment):
    """Remove columns of X, the same ones from the train and test sets."""

    name = "col_remove"
    title = "Remove columns"
    xlabel = "n_remove (columns)"

    def levels(self, dataset, n_levels):
        return np.linspace(0, dataset.n_features - 1, n_levels, dtype=int)

    def perturb(self, dataset, level, rng):
        n_features = dataset.n_features
        cols_kept = np.sort(rng.choice(n_features, size=n_features - level, replace=False))
        return dataset.perturbed(
            X_train=dataset.X_train[:, cols_kept], X_test=dataset.X_test[:, cols_kept]
        )


class RowRemove(Experiment):
    """Remove rows from the train set."""

    name = "row_remove"
    title = "Remove train rows"
    xlabel = "n_remove (rows)"

    def levels(self, dataset, n_levels):
        return np.linspace(0, dataset.n_train - 2, n_levels, dtype=int)

    def perturb(self, dataset, level, rng):
        n_train = dataset.n_train
        rows_kept = np.sort(rng.choice(n_train, size=n_train - level, replace=False))
        return dataset.perturbed(
            X_train=dataset.X_train[rows_kept], y_train=dataset.y_train[rows_kept]
        )


class RandomLabels(Experiment):
    """Replace train labels with random ones, drawn from the same classes."""

    name = "random_labels"
    title = "Randomize y_train"
    xlabel = "n_remove (labels replaced)"

    def levels(self, dataset, n_levels):
        return np.linspace(0, dataset.n_train, n_levels, dtype=int)

    def perturb(self, dataset, level, rng):
        label_classes = np.unique(dataset.y_train)
        y_random = dataset.y_train.copy()
        idx = rng.choice(dataset.n_train, size=level, replace=False)
        y_random[idx] = rng.choice(label_classes, size=level)
        return dataset.perturbed(y_train=y_random)


class ValueNoise(Experiment):
    """Replace individual X_train entries with noise matched to their own column."""

    name = "value_noise"
    title = "Randomize X values"
    xlabel = "n_remove (entries replaced)"

    def levels(self, dataset, n_levels):
        return np.linspace(0, dataset.n_train * dataset.n_features, n_levels, dtype=int)

    def perturb(self, dataset, level, rng):
        shape = (dataset.n_train, dataset.n_features)
        col_mean, col_std = dataset.X_train.mean(axis=0), dataset.X_train.std(axis=0)
        X_noisy = dataset.X_train.copy()
        flat_idx = rng.choice(shape[0] * shape[1], size=level, replace=False)
        rows_idx, cols_idx = np.unravel_index(flat_idx, shape)
        X_noisy[rows_idx, cols_idx] = rng.normal(col_mean[cols_idx], col_std[cols_idx])
        return dataset.perturbed(X_train=X_noisy)
