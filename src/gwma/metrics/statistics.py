from __future__ import annotations

from typing import Any

import numpy as np


def summarize(
    values: np.ndarray,
    seed: int = 42,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty array")
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled_means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "mean_ci95": [
            float(np.quantile(sampled_means, 0.025)),
            float(np.quantile(sampled_means, 0.975)),
        ],
    }
