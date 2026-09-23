"""How related a model's perturbed embeddings are to its unperturbed (baseline) ones.

Every metric here takes `(embeds, baseline)`, two (n_test, dim) arrays aligned row by row
(row i is the same test row in both), and returns one float, higher meaning more related.
They differ in what they need that alignment *for*:

- `cosine`: a per-row, deterministic measure - row i's cosine similarity only ever needs
  row i's two vectors, then the per-row values are averaged over the n_test rows.
- `mi` and `cka`: neither is defined for a single pair of vectors. Both instead treat the
  n_test rows as a sample from a joint distribution over (baseline, perturbed) and estimate
  a single number for the whole batch - there is no per-row value to average, the n_test
  rows are consumed together as the estimator's sample.

METRICS maps the --metric CLI choices to these functions.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_regression


def cosine_sim_to_baseline(embeds, baseline):
    """Cosine similarity between two sets of embeddings, computed on the last axis.

    Args:
        embeds (np.ndarray): The perturbed embeddings.
        baseline (np.ndarray): The unperturbed embeddings, same shape as `embeds`.

    Returns:
        np.ndarray: The per-sample cosine similarities.
    """
    embeds = np.asarray(embeds)
    baseline = np.asarray(baseline)
    num = np.sum(embeds * baseline, axis=-1)
    denom = np.linalg.norm(embeds, axis=-1) * np.linalg.norm(baseline, axis=-1)
    return num / denom


def cosine_metric(embeds, baseline):
    """The `cosine_sim_to_baseline` per-row values, reduced to one float by averaging."""
    return float(cosine_sim_to_baseline(embeds, baseline).mean())


# Embedding dimensions are typically in the dozens to hundreds; mi_metric fits one k-NN
# regressor per dimension, so its cost is O(dim) forward passes through sklearn's
# estimator. Past this many dimensions, a fixed, seeded random subset is used instead of
# all of them, to keep a run tractable - see mi_metric's docstring.
MI_MAX_DIMS = 64


def mi_metric(embeds, baseline, max_dims=MI_MAX_DIMS, seed=0):
    """Mutual information between baseline and perturbed embeddings, per dimension.

    A single pair of embedding vectors has no defined MI (unlike cosine similarity, which
    needs nothing beyond the two vectors themselves): mutual information is a property of
    a joint distribution, and estimating one needs a sample. The only sample available
    here is the n_test test rows, so this estimates MI one embedding dimension at a time
    - for dimension d, `baseline[:, d]` and `embeds[:, d]` (each of length n_test) are fed
    to a k-nearest-neighbour estimator (`sklearn.feature_selection.mutual_info_regression`,
    the Kraskov-Stogbauer-Grassberger estimator) - and averages the per-dimension MI.

    This is *not* the joint, multivariate MI between the two full embedding vectors (that
    would need a k-NN density estimate in the full embedding dimension, which the curse of
    dimensionality makes unreliable at the sample sizes n_test gives here, a few hundred to
    about a thousand rows against embeddings that can be as wide or wider); averaging
    per-dimension MI instead treats each dimension as independent, missing any dependence
    that only shows up jointly across dimensions, in exchange for an estimate that stays
    numerically meaningful at this sample size. `cka_metric` is the joint alternative.

    Args:
        embeds (np.ndarray): The perturbed embeddings, shape (n_test, dim).
        baseline (np.ndarray): The unperturbed embeddings, same shape as `embeds`.
        max_dims (int, optional): Past this many dimensions, a fixed, seeded random subset
            is used instead of every dimension, since cost scales with the dimension count
            used. Defaults to MI_MAX_DIMS.
        seed (int, optional): Seed of that subset, and of the estimator's own tie-breaking
            noise, so repeated calls on the same embeddings agree. Defaults to 0.

    Returns:
        float: The per-dimension MI, averaged over the dimensions used.
    """
    embeds = np.asarray(embeds).reshape(len(embeds), -1)
    baseline = np.asarray(baseline).reshape(len(baseline), -1)
    dim = baseline.shape[1]

    dims = range(dim)
    if dim > max_dims:
        dims = np.random.default_rng(seed).choice(dim, size=max_dims, replace=False)
        if not getattr(mi_metric, "_subset_note_shown", False):
            print(f"  mi: {dim} dims > max_dims={max_dims}, using a random {max_dims}-dim subset")
            mi_metric._subset_note_shown = True

    per_dim = [
        mutual_info_regression(baseline[:, [d]], embeds[:, d], random_state=seed)[0] for d in dims
    ]
    return float(np.mean(per_dim))


def _pairwise_sq_dists(X):
    """Squared Euclidean distance between every pair of rows of X, shape (n, n)."""
    sq_norms = np.sum(X**2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
    return np.clip(sq_dists, 0, None)  # float error can make a same-point distance slightly < 0


def _rbf_gram(X):
    """RBF Gram matrix of X, bandwidth set by the median heuristic (median pairwise
    distance of X itself), so it adapts to each embedding's own scale."""
    sq_dists = _pairwise_sq_dists(X)
    n = len(X)
    off_diag = sq_dists[~np.eye(n, dtype=bool)]
    sigma_sq = max(np.median(off_diag), 1e-12) / 2
    return np.exp(-sq_dists / (2 * sigma_sq))


def _center_gram(K):
    """`H @ K @ H` for the centering matrix H = I - 1/n, computed in O(n^2) not O(n^3)."""
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def _hsic(K, L):
    """Biased HSIC estimator of two (already Gram, not yet centered) matrices K, L."""
    n = K.shape[0]
    return float(np.sum(_center_gram(K) * _center_gram(L)) / (n - 1) ** 2)


def cka_metric(embeds, baseline):
    """Centered Kernel Alignment (CKA) between baseline and perturbed embeddings.

    Like `mi_metric`, a single pair of embedding vectors has no defined CKA: it needs a
    sample, here the n_test test rows, same as MI. Unlike MI though, it reads them jointly,
    across every dimension at once, rather than one dimension at a time - so it does not
    have the same curse-of-dimensionality problem MI does at these embedding widths. CKA is
    built on HSIC (the Hilbert-Schmidt Independence Criterion), a kernel-based dependence
    measure: it never estimates a density, only pairwise-kernel (here RBF, its bandwidth
    set from each embedding's own median pairwise distance) second-order statistics between
    the n_test x n_test Gram matrices of the two embeddings, which is why it stays reliable
    at far smaller sample-to-dimension ratios than MI's k-NN density estimate does. This is
    the same tool (Kornblith et al. 2019) used to compare neural network representations
    directly - exactly this project's baseline-vs-perturbed setting.

    Args:
        embeds (np.ndarray): The perturbed embeddings, shape (n_test, dim).
        baseline (np.ndarray): The unperturbed embeddings, same shape as `embeds`.

    Returns:
        float: The CKA similarity, in [0, 1].
    """
    embeds = np.asarray(embeds, dtype=np.float64).reshape(len(embeds), -1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(len(baseline), -1)

    K, L = _rbf_gram(baseline), _rbf_gram(embeds)
    hsic_xy, hsic_xx, hsic_yy = _hsic(K, L), _hsic(K, K), _hsic(L, L)
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 0 else 0.0


# The --metric CLI choices, each mapped to a (embeds, baseline) -> float callable.
METRICS = {"cosine": cosine_metric, "mi": mi_metric, "cka": cka_metric}

# What each metric's numbers mean, for the plot axis label.
METRIC_LABELS = {
    "cosine": "Cosine similarity to baseline",
    "mi": "Mutual information with baseline (nats, per dim)",
    "cka": "CKA similarity to baseline",
}
