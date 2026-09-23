import json
import os

import numpy as np


def permute_axis(X, axis=1, n_permut=1, rng=None):
    """Apply `n_permut` random pairwise swaps along one axis of an array.

    Args:
        X (np.ndarray): The array to permute.
        axis (int, optional): The axis along which slices are swapped. Defaults to 1.
        n_permut (int, optional): The number of pairwise swaps. Defaults to 1.
        rng (np.random.Generator, optional): The random generator. Defaults to None.

    Returns:
        np.ndarray: The permuted array.
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.moveaxis(np.array(X), axis, 0)
    size = X.shape[0]
    for _ in range(n_permut):
        i, j = rng.choice(size, size=2, replace=False)
        X[[i, j]] = X[[j, i]]
    return np.moveaxis(X, 0, axis)


def save_results(results, out_dir, file_name="results.json"):
    """Dump the experiment results as a json file.

    Args:
        results (dict): The results to save.
        out_dir (str): The output directory.
        file_name (str, optional): The name of the json file. Defaults to "results.json".

    Returns:
        str: The path of the written file.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, file_name)
    with open(path, "w") as file:
        json.dump(results, file, indent=2, default=_to_serializable)
    print(f"Saved results to {path}")
    return path


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)
