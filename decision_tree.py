# CAP 5610 - Group 10 | Decision Tree - Lydia Emmons

import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight


# load data
train_data = pd.read_csv("data/train_processed.csv")
test_data = pd.read_csv("data/test_processed.csv")
val_data = pd.read_csv("data/val_processed.csv")

# load features
train_features = sparse.load_npz("data/tfidf_train.npz")
test_features = sparse.load_npz("data/tfidf_test.npz")
val_features = sparse.load_npz("data/tfidf_val.npz")

# load labels
train_labels = train_data["label"]
test_labels = test_data["label"]
val_labels = val_data["label"]

emotion_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Random column subsampling per split: more candidates near the root, fewer deep
# (speed vs quality). Floor avoids tiny caps at deep nodes.
MAX_FEATURES_ROOT = 2048
MAX_FEATURES_FLOOR = 256


def max_features_for_depth(depth):
    return max(MAX_FEATURES_FLOOR, MAX_FEATURES_ROOT >> min(int(depth), 3))


def _split_rng(random_state, row_indices):
    rs = 0 if random_state is None else int(random_state)
    n = int(len(row_indices))
    lo = int(row_indices[0]) if n else 0
    hi = int(row_indices[-1]) if n else 0
    seed = (rs * 1_000_003 + n * 97 + lo * 11 + hi) % (2**32 - 1)
    return np.random.default_rng(seed)


class TreeNode:
    def __init__(self, feature=None, threshold=None, children=None, label=None, majority_label=None):
        self.feature = feature
        self.threshold = threshold
        self.children = children
        self.label = label
        self.majority_label = majority_label


def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0

    counts = Counter(labels)
    ent = 0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def majority_label(labels):
    if len(labels) == 0:
        return None
    return Counter(labels).most_common(1)[0][0]


def weighted_entropy(labels, weights):
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if labels.size == 0 or np.sum(weights) <= 0:
        return 0.0
    total_w = float(np.sum(weights))
    h = 0.0
    for c in np.unique(labels):
        w_c = float(np.sum(weights[labels == c]))
        if w_c > 0:
            p = w_c / total_w
            h -= p * math.log2(p)
    return h


def weighted_majority_class(labels, weights):
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if labels.size == 0:
        return None
    sums = {int(c): float(np.sum(weights[labels == c])) for c in np.unique(labels)}
    max_w = max(sums.values())
    tied = [c for c, w in sums.items() if abs(w - max_w) < 1e-15]
    return int(min(tied))


def best_split_sparse(X, y, row_indices, min_samples_leaf, random_state, sample_weights, depth):
    """
    Find the feature j and threshold t that maximize weighted information gain.
    Left branch: x[j] <= t. Candidate columns are subsampled (more at shallow depth).
    """
    row_indices = np.asarray(row_indices, dtype=np.intp)
    n = int(row_indices.size)
    if n < 2 * min_samples_leaf:
        return None

    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    y = np.asarray(y)
    labels_node = y[row_indices]
    weights_node = np.asarray(sample_weights[row_indices], dtype=np.float64)
    parent_h = weighted_entropy(labels_node, weights_node)

    sub = X[row_indices]
    cols = np.unique(sub.nonzero()[1])
    if cols.size == 0:
        return None

    cap = max_features_for_depth(depth)
    if cols.size > cap:
        rng = _split_rng(random_state, row_indices)
        cols = rng.choice(cols, size=cap, replace=False)

    best_gain = 0.0
    best_j = None
    best_thr = None

    for j in cols:
        vals = np.asarray(X[row_indices, j].todense()).ravel()
        order = np.argsort(vals, kind="mergesort")
        vals_s = vals[order]
        lab_s = labels_node[order]
        w_s = weights_node[order]

        for i in range(n - 1):
            if vals_s[i] == vals_s[i + 1]:
                continue
            left_n = i + 1
            right_n = n - left_n
            if left_n < min_samples_leaf or right_n < min_samples_leaf:
                continue
            thr = (vals_s[i] + vals_s[i + 1]) / 2.0
            left_h = weighted_entropy(lab_s[:left_n], w_s[:left_n])
            right_h = weighted_entropy(lab_s[left_n:], w_s[left_n:])
            w_left = float(np.sum(w_s[:left_n]))
            w_right = float(np.sum(w_s[left_n:]))
            w_tot = w_left + w_right
            if w_tot <= 0:
                continue
            gain = parent_h - ((w_left / w_tot) * left_h + (w_right / w_tot) * right_h)
            if gain > best_gain:
                best_gain = gain
                best_j = int(j)
                best_thr = float(thr)

    if best_j is None:
        return None
    return best_j, best_thr


def build_tree(X, y, row_indices, depth, max_depth, min_samples_split, min_samples_leaf, random_state, sample_weights):
    row_indices = np.asarray(row_indices, dtype=np.intp)
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()

    labels_node = np.asarray(y[row_indices])
    sw_node = np.asarray(sample_weights[row_indices], dtype=np.float64)

    if labels_node.size < min_samples_split:
        pred = weighted_majority_class(labels_node, sw_node)
        return TreeNode(label=pred, majority_label=pred)

    elif depth >= max_depth:
        pred = weighted_majority_class(labels_node, sw_node)
        return TreeNode(label=pred, majority_label=pred)

    elif np.unique(labels_node).size == 1:
        pred = int(labels_node[0])
        return TreeNode(label=pred, majority_label=pred)

    split = best_split_sparse(X, y, row_indices, min_samples_leaf, random_state, sample_weights, depth)
    if split is None:
        pred = weighted_majority_class(labels_node, sw_node)
        return TreeNode(label=pred, majority_label=pred)

    j, t = split
    vals = np.asarray(X[row_indices, j].todense()).ravel()
    left_idx = row_indices[vals <= t]
    right_idx = row_indices[vals > t]

    maj = weighted_majority_class(labels_node, sw_node)
    left_child = build_tree(
        X, y, left_idx, depth + 1, max_depth, min_samples_split, min_samples_leaf, random_state, sample_weights
    )
    right_child = build_tree(
        X, y, right_idx, depth + 1, max_depth, min_samples_split, min_samples_leaf, random_state, sample_weights
    )

    return TreeNode(
        feature=j,
        threshold=t,
        children={"left": left_child, "right": right_child},
        majority_label=maj,
    )


def _feature_value(x, j):
    """Single scalar x[j] for one sample: 1-D array or one row (dense/sparse)."""
    if sparse.issparse(x):
        x = x.tocsr()
        return float(x[0, j])
    a = np.asarray(x)
    if a.ndim == 2:
        return float(a[0, j])
    return float(a[j])


def predict_one(node, x):
    current = node
    while current.label is None:
        j = current.feature
        t = current.threshold
        children = current.children
        if children is None:
            return current.majority_label
        v = _feature_value(x, j)
        nxt = children["left"] if v <= t else children["right"]
        if nxt is None:
            return current.majority_label
        current = nxt
    return current.label


def predict_all(node, X):
    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)
    n = X.shape[0]
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        row = X[i] if sparse.issparse(X) else X[i : i + 1]
        out[i] = predict_one(node, row)
    return out


def train_decision_tree(
    X_train,
    y_train,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    random_state,
    class_weight="balanced",
):
    y_train = np.asarray(y_train)
    if sparse.issparse(X_train):
        X_train = X_train.tocsr()

    n_samples = X_train.shape[0]
    row_indices = np.arange(n_samples, dtype=np.intp)
    if class_weight == "balanced":
        sample_weights = compute_sample_weight("balanced", y_train).astype(np.float64)
    else:
        sample_weights = np.ones(n_samples, dtype=np.float64)

    return build_tree(
        X_train,
        y_train,
        row_indices,
        0,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        random_state,
        sample_weights,
    )


def accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)

    for i in range(total):
        if y_true[i] == y_pred[i]:
            correct += 1

    if total == 0:
        return 0

    accuracy = correct / total

    return accuracy


def _print_metrics(split_name, y_true, y_pred):
    acc = accuracy(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(
        f"{split_name} — accuracy: {acc:.4f}, "
        f"macro precision: {macro_p:.4f}, macro recall: {macro_r:.4f}, macro F1: {macro_f1:.4f}"
    )


def tune_hyperparameters(X_train, y_train, X_val, y_val, random_state):
    """Grid search on validation macro-F1 (suggestion 3)."""
    X_val = X_val.tocsr() if sparse.issparse(X_val) else np.asarray(X_val)
    y_val = np.asarray(y_val)

    grid = []
    for max_depth in (12, 18, 24):
        for min_samples_leaf, min_samples_split in ((4, 10), (4, 20), (8, 16), (8, 20)):
            grid.append((max_depth, min_samples_leaf, min_samples_split))

    best_f1 = -1.0
    best_params = grid[0]
    for k, (max_depth, min_samples_leaf, min_samples_split) in enumerate(grid, start=1):
        root = train_decision_tree(
            X_train,
            y_train,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight="balanced",
        )
        pred_val = predict_all(root, X_val)
        f1v = f1_score(y_val, pred_val, average="macro", zero_division=0)
        print(
            f"  [{k}/{len(grid)}] depth={max_depth} min_split={min_samples_split} "
            f"min_leaf={min_samples_leaf} -> val macro F1={f1v:.4f}"
        )
        if f1v > best_f1:
            best_f1 = f1v
            best_params = (max_depth, min_samples_leaf, min_samples_split)

    return {"macro_f1": best_f1, "params": best_params}


def main():
    random_state = 42
    tune_on_validation = True
    # Use first N training rows only (set to None to use the full training set).
    train_subset_n = 20_000

    y_train_arr = np.asarray(train_labels)
    y_val_arr = np.asarray(val_labels)
    y_test_arr = np.asarray(test_labels)

    if train_subset_n is not None:
        n = min(int(train_subset_n), train_features.shape[0])
        X_train_fit = train_features[:n].tocsr()
        y_train_fit = y_train_arr[:n]
    else:
        n = train_features.shape[0]
        X_train_fit = train_features.tocsr()
        y_train_fit = y_train_arr

    req = "full train set" if train_subset_n is None else f"up to {train_subset_n} rows"
    print(
        f"train_features.shape={train_features.shape}, "
        f"fitting on n_rows={n} (requested {req}), "
        f"class_weight=balanced, max_features(depth)= "
        f"{max_features_for_depth(0)}..{MAX_FEATURES_FLOOR}"
    )

    if tune_on_validation:
        print("Validation grid search (macro F1)…")
        best = tune_hyperparameters(X_train_fit, y_train_fit, val_features, y_val_arr, random_state)
        max_depth, min_samples_leaf, min_samples_split = best["params"]
        print(
            f"Best validation macro F1={best['macro_f1']:.4f} with "
            f"max_depth={max_depth}, min_samples_split={min_samples_split}, "
            f"min_samples_leaf={min_samples_leaf}"
        )
    else:
        max_depth, min_samples_leaf, min_samples_split = 18, 8, 20

    root = train_decision_tree(
        X_train_fit,
        y_train_fit,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight="balanced",
    )

    pred_train = predict_all(root, X_train_fit)
    pred_val = predict_all(root, val_features)
    pred_test = predict_all(root, test_features)

    _print_metrics("Train (subset used for fitting)" if train_subset_n else "Train", y_train_fit, pred_train)
    _print_metrics("Validation", y_val_arr, pred_val)
    _print_metrics("Test", y_test_arr, pred_test)

    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test_arr,
        pred_test,
        display_labels=emotion_names,
        cmap="Blues",
        ax=ax,
        colorbar=True,
    )
    ax.set_title("Decision Tree - Confusion Matrix")
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("True Emotion")
    plt.tight_layout()
    plt.savefig("results/decision_tree_confusion_matrix.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
