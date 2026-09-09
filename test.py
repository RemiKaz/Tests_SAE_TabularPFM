from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
 
from tabpfn import TabPFNClassifier
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def permute_axis(X, axis=1, n_permut=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    X = np.moveaxis(np.array(X), axis, 0)
    size = X.shape[0]
    for _ in range(n_permut):
        i, j = rng.choice(size, size=2, replace=False)
        X[[i, j]] = X[[j, i]]
    return np.moveaxis(X, 0, axis)


def cosine_sim_to_baseline(embeds, baseline):
    embeds = np.asarray(embeds)
    baseline = np.asarray(baseline)
    num = np.sum(embeds * baseline, axis=-1)
    denom = np.linalg.norm(embeds, axis=-1) * np.linalg.norm(baseline, axis=-1)
    return num / denom


# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize the classifier
clf = TabPFNClassifier(model_path="/lustre/fswork/projects/rech/wqn/ufb58bn/hf_cache/hub/models--Prior-Labs--tabpfn_3/snapshots/24a16a89d245878b846555110985634aa2e656d7/tabpfn-v3-classifier-v3_default.ckpt",
                       device="cuda")

rng = np.random.default_rng(0)
n_train, n_features = X_train.shape
n_levels = 20  # number of intensity steps swept per perturbation type

# Baseline: no perturbation
clf.fit(X_train, y_train)
embeds_baseline = clf.get_embeddings(X_test)


def mean_sim(embeds):
    return cosine_sim_to_baseline(embeds, embeds_baseline).mean()


# 1. Shuffle rows in X_train and y_train (jointly, so pairing is preserved),
#    intensity = number of pairwise row swaps
row_shuffle_levels = np.linspace(0, n_train, n_levels, dtype=int)
sim_row_shuf = []
for k in row_shuffle_levels:
    train_row_shuf = permute_axis(np.column_stack([X_train, y_train]), axis=0, n_permut=int(k), rng=rng)
    Xk, yk = train_row_shuf[:, :-1], train_row_shuf[:, -1].astype(y_train.dtype)
    clf.fit(Xk, yk)
    sim_row_shuf.append(mean_sim(clf.get_embeddings(X_test)))

# 2. Shuffle columns in X (same column order applied to train and test),
#    intensity = number of pairwise column swaps
col_shuffle_levels = np.linspace(0, n_features, n_levels, dtype=int)
sim_col_shuf = []
for k in col_shuffle_levels:
    X_col_shuf = permute_axis(np.vstack([X_train, X_test]), axis=1, n_permut=int(k), rng=rng)
    Xk_train, Xk_test = X_col_shuf[:n_train], X_col_shuf[n_train:]
    clf.fit(Xk_train, y_train)
    sim_col_shuf.append(mean_sim(clf.get_embeddings(Xk_test)))

# 3. Remove columns in X (same columns dropped from train and test),
#    intensity = number of columns removed
col_remove_levels = np.linspace(0, n_features - 1, n_levels, dtype=int)
sim_col_removed = []
for k in col_remove_levels:
    cols_kept = np.sort(rng.choice(n_features, size=n_features - int(k), replace=False))
    clf.fit(X_train[:, cols_kept], y_train)
    sim_col_removed.append(mean_sim(clf.get_embeddings(X_test[:, cols_kept])))

# 4. Remove rows in X_train and y_train,
#    intensity = number of training rows removed
row_remove_levels = np.linspace(0, n_train - 2, n_levels, dtype=int)
sim_row_removed = []
for k in row_remove_levels:
    rows_kept = np.sort(rng.choice(n_train, size=n_train - int(k), replace=False))
    clf.fit(X_train[rows_kept], y_train[rows_kept])
    sim_row_removed.append(mean_sim(clf.get_embeddings(X_test)))

# 5. Replace y_train labels with random labels (drawn from the same classes),
#    intensity = number of training labels replaced
label_classes = np.unique(y_train)
random_label_levels = np.linspace(0, n_train, n_levels, dtype=int)
sim_random_labels = []
for k in random_label_levels:
    y_random = y_train.copy()
    idx = rng.choice(n_train, size=int(k), replace=False)
    y_random[idx] = rng.choice(label_classes, size=int(k))
    clf.fit(X_train, y_random)
    sim_random_labels.append(mean_sim(clf.get_embeddings(X_test)))

# 6. Replace individual values in X_train with random noise (matched to each
#    column's own mean/std), intensity = number of (row, column) entries replaced
col_mean, col_std = X_train.mean(axis=0), X_train.std(axis=0)
n_entries = n_train * n_features
value_noise_levels = np.linspace(0, n_entries, n_levels, dtype=int)
sim_value_noise = []
for k in value_noise_levels:
    X_noisy = X_train.copy()
    flat_idx = rng.choice(n_entries, size=int(k), replace=False)
    rows_idx, cols_idx = np.unravel_index(flat_idx, (n_train, n_features))
    X_noisy[rows_idx, cols_idx] = rng.normal(col_mean[cols_idx], col_std[cols_idx])
    clf.fit(X_noisy, y_train)
    sim_value_noise.append(mean_sim(clf.get_embeddings(X_test)))

# Plot cosine similarity to baseline vs. perturbation intensity
ink = "#0b0b0b"
muted = "#898781"
grid_color = "#e1e0d9"
series_color = "#2a78d6"

fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="#fcfcfb")
panels = [
    (axes[0, 0], row_shuffle_levels, sim_row_shuf, "Shuffle train rows", "n_permut (row swaps)"),
    (axes[0, 1], col_shuffle_levels, sim_col_shuf, "Shuffle columns", "n_permut (column swaps)"),
    (axes[0, 2], random_label_levels, sim_random_labels, "Randomize y_train", "n_remove (labels replaced)"),
    (axes[1, 0], col_remove_levels, sim_col_removed, "Remove columns", "n_remove (columns)"),
    (axes[1, 1], row_remove_levels, sim_row_removed, "Remove train rows", "n_remove (rows)"),
    (axes[1, 2], value_noise_levels, sim_value_noise, "Randomize X values", "n_remove (entries replaced)"),
]
for ax, levels, sims, title, xlabel in panels:
    ax.set_facecolor("#fcfcfb")
    ax.plot(levels, sims, color=series_color, linewidth=2, marker="o", markersize=6)
    ax.set_title(title, color=ink, fontsize=11)
    ax.set_xlabel(xlabel, color=muted, fontsize=9)
    ax.set_ylabel("Cosine similarity to baseline", color=muted, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, color=grid_color, linewidth=0.8)
    ax.tick_params(colors=muted, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(grid_color)

fig.tight_layout()
fig.savefig("perturbation_invariance.png", dpi=150, facecolor=fig.get_facecolor())
print("Saved plot to perturbation_invariance.png")

# 7. Predict a different column of X instead of the original label
#    (binarized via a median split), with that column dropped from the
#    features; x-axis = which column of X was used as the alternative target
sim_alt_target = []
for c in range(n_features):
    y_alt = (X[:, c] > np.median(X[:, c])).astype(int)
    X_alt = np.delete(X, c, axis=1)
    X_alt_train, X_alt_test, y_alt_train, _ = train_test_split(
        X_alt, y_alt, test_size=0.33, random_state=42
    )
    clf.fit(X_alt_train, y_alt_train)
    sim_alt_target.append(mean_sim(clf.get_embeddings(X_alt_test)))

fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
ax2.set_facecolor("#fcfcfb")
ax2.plot(range(n_features), sim_alt_target, color=series_color, linewidth=2, marker="o", markersize=6)
ax2.set_title("Predict a different column of X instead of y", color=ink, fontsize=11)
ax2.set_xlabel("Column index used as alternative target", color=muted, fontsize=9)
ax2.set_ylabel("Cosine similarity to baseline", color=muted, fontsize=9)
ax2.set_ylim(0, 1.05)
ax2.grid(True, color=grid_color, linewidth=0.8)
ax2.tick_params(colors=muted, labelsize=8)
for spine in ("top", "right"):
    ax2.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax2.spines[spine].set_color(grid_color)
fig2.tight_layout()
fig2.savefig("target_column_invariance.png", dpi=150, facecolor=fig2.get_facecolor())
print("Saved plot to target_column_invariance.png")

# Refit on the unperturbed data for the rest of the pipeline below
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)

# prediction_probabilities type
print(type(prediction_probabilities))

print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))
 
# Predict class labels
predictions = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))