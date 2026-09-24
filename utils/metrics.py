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


def _as_rows(embeds, baseline, name):
    """Both embeddings as (n_test, dim) float arrays, refusing shapes the sample-based
    metrics would silently get wrong (e.g. a leading ensemble axis read as one sample)."""
    embeds, baseline = np.asarray(embeds, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if embeds.shape != baseline.shape or embeds.ndim != 2:
        raise ValueError(
            f"{name} needs two (n_test, dim) embeddings of the same shape, got "
            f"{embeds.shape} and {baseline.shape}"
        )
    if len(embeds) < 4:
        raise ValueError(f"{name} needs at least 4 test rows as its sample, got {len(embeds)}")
    return embeds, baseline


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


def ccos_metric(embeds, baseline):
    """Centered cosine: cosine similarity after subtracting the baseline's mean embedding
    from both, per test row, then averaged.

    Transformer layers often give every sample a large shared component (the same mean
    vector for all rows). Plain cosine is dominated by it: a layer with a big shared mean
    reads as nearly unchanged whatever happens to the part that differs between samples,
    so comparing cosine across layers can invert their order (a layer with a shared mean
    and more damage read as more invariant than a centered one with less, in every
    synthetic trial). Removing the baseline mean, the same reference for both embeddings,
    measures how each sample's own position moved, which is what differs between layers.
    """
    embeds, baseline = _as_rows(embeds, baseline, "ccos")
    mean = baseline.mean(axis=0)
    return cosine_metric(embeds - mean, baseline - mean)


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
    embeds, baseline = _as_rows(embeds, baseline, "mi")
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


def nmi_metric(embeds, baseline, max_dims=MI_MAX_DIMS, seed=0):
    """`mi_metric` divided by the baseline's MI with itself, per dimension: the fraction of
    each dimension's information the perturbed embedding still carries, 1 when unchanged.

    The k-NN estimator cannot return an infinite MI for a variable against itself: it tops
    out near log(n_test), so raw `mi` values have a ceiling set by the test-set size (about
    2.6 nats on iris against 3.6 on digits), not by the embeddings. Dividing each dimension
    by that ceiling, measured on the baseline itself, removes it, so curves can be compared
    across datasets. Dimensions constant over the test set carry no information and are
    left out (their ratio is undefined).

    Args:
        embeds (np.ndarray): The perturbed embeddings, shape (n_test, dim).
        baseline (np.ndarray): The unperturbed embeddings, same shape as `embeds`.
        max_dims (int, optional): As in `mi_metric`. Defaults to MI_MAX_DIMS.
        seed (int, optional): As in `mi_metric`. Defaults to 0.

    Returns:
        float: The per-dimension normalized MI, averaged over the informative dimensions.
    """
    embeds, baseline = _as_rows(embeds, baseline, "nmi")
    dim = baseline.shape[1]
    dims = np.arange(dim)
    if dim > max_dims:
        dims = np.random.default_rng(seed).choice(dim, size=max_dims, replace=False)

    ratios = []
    for d in dims:
        ceiling = _self_mi(baseline[:, d], seed)
        if ceiling <= 0:
            continue
        mi = mutual_info_regression(baseline[:, [d]], embeds[:, d], random_state=seed)[0]
        ratios.append(mi / ceiling)
    return float(np.mean(ratios)) if ratios else 0.0


# The baseline is the same at every level of an experiment, so its self-MI is cached.
_SELF_MI_CACHE = {}


def _self_mi(column, seed):
    key = (column.tobytes(), seed)
    if key not in _SELF_MI_CACHE:
        if len(_SELF_MI_CACHE) > 100_000:
            _SELF_MI_CACHE.clear()
        _SELF_MI_CACHE[key] = mutual_info_regression(column[:, None], column, random_state=seed)[0]
    return _SELF_MI_CACHE[key]


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


def _hsic_unbiased(K, L):
    """Unbiased HSIC estimator (Song et al. 2012) of two Gram matrices K, L."""
    n = K.shape[0]
    K, L = K.copy(), L.copy()
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    ones = np.ones(n)
    Lsum_rows = L @ ones
    term = np.sum(K * L) + (K.sum() * L.sum()) / ((n - 1) * (n - 2)) - 2 / (n - 2) * (ones @ K @ Lsum_rows)
    return float(term / (n * (n - 3)))


def dcka_metric(embeds, baseline):
    """Debiased CKA: `cka_metric` with the unbiased HSIC estimator in place of the biased one.

    The biased estimator behind `cka_metric` reads two completely independent embeddings
    as similar when the embedding is wide against the number of test rows (independent
    Gaussian embeddings: 0.78 at 50 rows x 64 dims, 0.99 at 188 rows x 4096 dims), so its
    floor changes from one dataset and layer to the next. The unbiased estimator removes
    that: independent embeddings read about 0 at any size, identical ones still exactly 1.
    It can come out slightly negative (estimation noise around 0). This is the "debiased
    CKA" of Kornblith et al.'s own code (Nguyen et al. 2021).

    Args:
        embeds (np.ndarray): The perturbed embeddings, shape (n_test, dim).
        baseline (np.ndarray): The unperturbed embeddings, same shape as `embeds`.

    Returns:
        float: The debiased CKA similarity, about 0 for independent, 1 for identical.
    """
    embeds, baseline = _as_rows(embeds, baseline, "dcka")
    K, L = _rbf_gram(baseline), _rbf_gram(embeds)
    hsic_xy, hsic_xx, hsic_yy = _hsic_unbiased(K, L), _hsic_unbiased(K, K), _hsic_unbiased(L, L)
    if hsic_xx <= 0 or hsic_yy <= 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


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
    embeds, baseline = _as_rows(embeds, baseline, "cka")

    K, L = _rbf_gram(baseline), _rbf_gram(embeds)
    hsic_xy, hsic_xx, hsic_yy = _hsic(K, L), _hsic(K, K), _hsic(L, L)
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 0 else 0.0


# The --metric CLI choices, each mapped to a (embeds, baseline) -> float callable.
METRICS = {
    "cosine": cosine_metric,
    "ccos": ccos_metric,
    "mi": mi_metric,
    "nmi": nmi_metric,
    "cka": cka_metric,
    "dcka": dcka_metric,
}

# What each metric's numbers mean, for the plot axis label.
METRIC_LABELS = {
    "cosine": "Cosine similarity to baseline",
    "mi": "Mutual information with baseline (nats, per dim)",
    "ccos": "Centered cosine similarity to baseline",
    "nmi": "Normalized mutual information with baseline",
    "cka": "CKA similarity to baseline (biased)",
    "dcka": "Debiased CKA similarity to baseline",
}
