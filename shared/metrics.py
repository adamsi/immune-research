"""Evaluation metrics shared across projects."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auc_with_ci(y, prob, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """AUROC with a 95% percentile bootstrap confidence interval over patients."""
    y, prob = np.asarray(y), np.asarray(prob)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        if len(set(y[s])) > 1:
            boot.append(roc_auc_score(y[s], prob[s]))
    return roc_auc_score(y, prob), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def macro_auc_with_ci(y, prob, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """Macro one-vs-rest AUROC for a multi-class problem (prob: n x classes), with a 95% bootstrap CI.
    Resamples missing a class are skipped."""
    y, prob = np.asarray(y), np.asarray(prob)
    labels = np.arange(prob.shape[1])
    score = lambda yy, pp: roc_auc_score(yy, pp, multi_class="ovr", average="macro", labels=labels)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        if len(set(y[s])) == len(labels):
            boot.append(score(y[s], prob[s]))
    return score(y, prob), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
