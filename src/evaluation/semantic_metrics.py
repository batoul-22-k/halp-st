from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(labels, cosine_scores) -> dict:
    labels = np.asarray(labels, dtype=float)
    cosine_scores = np.asarray(cosine_scores, dtype=float)
    normalized_predictions = (cosine_scores + 1.0) / 2.0
    mse = mean_squared_error(labels, normalized_predictions)
    return {
        "spearman": float(spearmanr(labels, cosine_scores).correlation),
        "pearson": float(pearsonr(labels, cosine_scores).statistic),
        "mse": float(mse),
        "mae": float(mean_absolute_error(labels, normalized_predictions)),
        "rmse": float(np.sqrt(mse)),
    }


def bootstrap_spearman_ci(labels, cosine_scores, n_bootstrap: int = 1000, seed: int = 42, confidence: float = 0.95) -> dict:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    scores = np.asarray(cosine_scores)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(labels), len(labels))
        corr = spearmanr(labels[idx], scores[idx]).correlation
        if np.isfinite(corr):
            values.append(corr)
    alpha = (1.0 - confidence) / 2.0
    return {
        "spearman_ci_low": float(np.quantile(values, alpha)),
        "spearman_ci_high": float(np.quantile(values, 1.0 - alpha)),
    }

