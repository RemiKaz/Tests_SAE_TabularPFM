from dataclasses import dataclass, replace

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class TabularDataset:
    """A tabular dataset, already split in a train and a test set.

    `X` and `y` keep the unsplit data, so that an experiment can re-split the
    dataset with a different target column (see `experiments.target_column`).

    Attributes:
        name (str): The name of the dataset.
        X (np.ndarray): The full features, of shape (n_samples, n_features).
        y (np.ndarray): The full labels, of shape (n_samples,).
        X_train (np.ndarray): The train features.
        y_train (np.ndarray): The train labels.
        X_test (np.ndarray): The test features.
        y_test (np.ndarray): The test labels.
        test_size (float): The proportion of the dataset used as test set.
        seed (int): The seed of the train/test split.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_size: float
    seed: int

    @classmethod
    def from_xy(cls, name, X, y, test_size=0.33, seed=42):
        """Build a dataset by splitting `X` and `y` in a train and a test set.

        Args:
            name (str): The name of the dataset.
            X (np.ndarray): The features.
            y (np.ndarray): The labels.
            test_size (float, optional): The test proportion. Defaults to 0.33.
            seed (int, optional): The seed of the split. Defaults to 42.

        Returns:
            TabularDataset: The split dataset.
        """
        X, y = np.asarray(X), np.asarray(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        return cls(
            name=name,
            X=X,
            y=y,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            test_size=test_size,
            seed=seed,
        )

    @property
    def n_train(self):
        return self.X_train.shape[0]

    @property
    def n_test(self):
        return self.X_test.shape[0]

    @property
    def n_features(self):
        return self.X_train.shape[1]

    def perturbed(self, **splits):
        """Return a copy of the dataset with some of the splits replaced.

        Args:
            splits (dict): The fields to override, in [X_train, y_train, X_test, y_test].

        Returns:
            TabularDataset: The perturbed dataset.
        """
        return replace(self, **splits)
