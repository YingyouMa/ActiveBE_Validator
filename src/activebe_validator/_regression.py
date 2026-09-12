"""Internal regression helpers shared by equation validators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pysindy as ps


def fit_linear_term_model(
    target: np.ndarray,
    features: Sequence[np.ndarray],
    *,
    threshold: float,
    alpha: float,
    normalize_columns: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Fit a sparse linear combination of candidate equation terms."""
    y = target.reshape(-1)
    x = np.column_stack([feature.reshape(-1) for feature in features])

    active_mask = np.ones(x.shape[1], dtype=bool)
    if normalize_columns:
        column_norms = np.linalg.norm(x, axis=0)
        active_mask = column_norms > 0.0

    x_active = x[:, active_mask]
    if x_active.shape[1] == 0:
        coefficients = np.zeros(x.shape[1], dtype=float)
        prediction = np.zeros_like(y)
    else:
        optimizer = ps.STLSQ(
            threshold=threshold,
            alpha=alpha,
            normalize_columns=normalize_columns,
            unbias=True,
        )
        optimizer.fit(x_active, y)
        active_coefficients = np.asarray(optimizer.coef_, dtype=float).reshape(-1)
        coefficients = np.zeros(x.shape[1], dtype=float)
        coefficients[active_mask] = active_coefficients
        prediction = np.asarray(optimizer.predict(x_active), dtype=float).reshape(-1)

    residual = y - prediction
    residual_norm = np.linalg.norm(residual)
    target_norm = np.linalg.norm(y)
    rmse = float(np.sqrt(np.mean(residual**2)))
    relative_residual = float(residual_norm / target_norm) if target_norm > 0 else 0.0

    centered_target = y - np.mean(y)
    denominator = float(np.sum(centered_target**2))
    if denominator > 0.0:
        r2 = float(1.0 - np.sum(residual**2) / denominator)
    else:
        r2 = 1.0 if residual_norm == 0.0 else 0.0

    return coefficients, prediction, r2, rmse, relative_residual
