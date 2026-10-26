"""PR-AUC, lift (PR-AUC / late rate), ROC-AUC, precision/recall in the top 10%, Brier, ECE."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from opspilot.config import RANDOM_SEED

# The business question: if an operations team can check 1 order in 10, which ones?
TOP_FRACTION = 0.10


def pr_auc(y_true, p) -> float:
    return float(average_precision_score(y_true, p))


def lift(y_true, p) -> float:
    """PR-AUC divided by the late rate. A random model scores 1."""
    return pr_auc(y_true, p) / float(np.mean(y_true))


def roc_auc(y_true, p) -> float:
    return float(roc_auc_score(y_true, p))


def precision_recall_at_top(y_true, p, fraction: float = TOP_FRACTION) -> tuple[float, float]:
    """Precision and recall among the `fraction` of orders with the highest scores.

    Orders tied at the cut-off are shared out pro rata, so a model that gives
    everyone the same score gets the late rate, whatever order the rows are in.
    """
    y_true, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    k = int(np.ceil(fraction * len(p)))
    threshold = np.sort(p)[::-1][k - 1]
    above, tied = p > threshold, p == threshold
    caught = y_true[above].sum() + (k - above.sum()) * y_true[tied].mean()
    return float(caught / k), float(caught / y_true.sum())


def brier(y_true, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def expected_calibration_error(y_true, p, n_bins: int = 10) -> float:
    """Gap between predicted and observed late rate, averaged over equal-width bins."""
    y_true, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    gaps = [
        np.mean(bins == b) * abs(p[bins == b].mean() - y_true[bins == b].mean())
        for b in np.unique(bins)
    ]
    return float(np.sum(gaps))


def evaluate(y_true, p) -> dict[str, float]:
    """Every metric the project reports, for one set of predictions."""
    precision, recall = precision_recall_at_top(y_true, p)
    return {
        "n": float(len(y_true)),
        "late_rate": float(np.mean(y_true)),
        "pr_auc": pr_auc(y_true, p),
        "lift": lift(y_true, p),
        "roc_auc": roc_auc(y_true, p),
        "precision_top10": precision,
        "recall_top10": recall,
        "brier": brier(y_true, p),
        "ece": expected_calibration_error(y_true, p),
    }


def paired_bootstrap(
    y_true, p_a, p_b, metric=lift, n_resamples: int = 1000, seed: int = RANDOM_SEED
) -> tuple[float, float, float]:
    """metric(a) minus metric(b), with a 95% interval from resampling the same orders for both."""
    y_true, p_a, p_b = (np.asarray(x, dtype=float) for x in (y_true, p_a, p_b))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_resamples):
        i = rng.integers(0, len(y_true), len(y_true))
        if 0 < y_true[i].sum() < len(i):
            differences.append(metric(y_true[i], p_a[i]) - metric(y_true[i], p_b[i]))
    low, high = np.percentile(differences, [2.5, 97.5])
    return metric(y_true, p_a) - metric(y_true, p_b), float(low), float(high)


def bootstrap_interval(
    y_true, p, metric=pr_auc, n_resamples: int = 1000, seed: int = RANDOM_SEED
) -> tuple[float, float]:
    """95% interval of a metric from resampling the scored orders."""
    y_true, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        i = rng.integers(0, len(y_true), len(y_true))
        if 0 < y_true[i].sum() < len(i):
            values.append(metric(y_true[i], p[i]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def recall_top10(y_true, p) -> float:
    return precision_recall_at_top(y_true, p)[1]
