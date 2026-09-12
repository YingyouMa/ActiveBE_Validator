"""Residual evaluation helpers for pointwise SINDy-style equation checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointwiseResidualEvaluationResult:
    """Residual summary for a fixed pointwise feature model and coefficient set."""

    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    coefficient_by_name: dict[str, float]
    target: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    sample_count: int
    residual_norm: float
    target_norm: float
    relative_residual: float
    rmse: float
    max_abs_residual: float
    mean_residual: float


def _coefficient_vector_from_input(
    feature_names: Sequence[str],
    coefficients: Sequence[float] | np.ndarray | Mapping[str, float],
) -> np.ndarray:
    feature_names = tuple(feature_names)
    if isinstance(coefficients, Mapping):
        unknown_names = sorted(set(coefficients) - set(feature_names))
        if unknown_names:
            valid_names = ", ".join(feature_names)
            raise ValueError(
                f"Unknown coefficient names {unknown_names}; valid feature names are: {valid_names}."
            )
        return np.asarray(
            [float(coefficients.get(name, 0.0)) for name in feature_names],
            dtype=float,
        )

    coefficient_array = np.asarray(coefficients, dtype=float).reshape(-1)
    if coefficient_array.size != len(feature_names):
        raise ValueError(
            "Coefficient array length must match the number of features; got "
            f"{coefficient_array.size} coefficients for {len(feature_names)} features."
        )
    return coefficient_array


def evaluate_feature_model_residual(
    target: np.ndarray,
    features: np.ndarray,
    feature_names: Sequence[str],
    coefficients: Sequence[float] | np.ndarray | Mapping[str, float],
) -> PointwiseResidualEvaluationResult:
    """Evaluate a fixed coefficient vector against a pointwise feature tensor.

    Parameters
    ----------
    target:
        Left-hand-side data array, for example ``dt(v)`` or ``dt(Q)`` after the
        desired time discretization and spatial sampling have already been applied.
    features:
        Feature tensor whose last axis indexes the candidate right-hand-side terms.
        Its shape must be ``target.shape + (n_features,)``.
    feature_names:
        Names corresponding to the last axis of ``features``.
    coefficients:
        Either a dense coefficient vector ordered like ``feature_names`` or a
        mapping from feature name to coefficient value. Missing mapping entries
        default to zero.
    """

    feature_names = tuple(feature_names)
    feature_array = np.asarray(features, dtype=float)
    target_array = np.asarray(target, dtype=float)

    if feature_array.ndim < 1:
        raise ValueError("features must have at least one dimension.")
    if feature_array.shape[:-1] != target_array.shape:
        raise ValueError(
            "features must have shape target.shape + (n_features,); got "
            f"target shape {target_array.shape} and feature shape {feature_array.shape}."
        )
    if feature_array.shape[-1] != len(feature_names):
        raise ValueError(
            "The last feature axis must match feature_names; got "
            f"{feature_array.shape[-1]} feature columns and {len(feature_names)} names."
        )

    coefficient_vector = _coefficient_vector_from_input(feature_names, coefficients)
    prediction = np.tensordot(feature_array, coefficient_vector, axes=([-1], [0]))
    residual = target_array - prediction

    residual_norm = float(np.linalg.norm(residual.ravel()))
    target_norm = float(np.linalg.norm(target_array.ravel()))
    rmse = float(np.sqrt(np.mean(residual**2)))
    relative_residual = residual_norm / target_norm if target_norm > 0.0 else 0.0

    return PointwiseResidualEvaluationResult(
        feature_names=feature_names,
        coefficients=coefficient_vector,
        coefficient_by_name=dict(zip(feature_names, coefficient_vector)),
        target=target_array,
        prediction=np.asarray(prediction, dtype=float),
        residual=np.asarray(residual, dtype=float),
        sample_count=int(target_array.size),
        residual_norm=residual_norm,
        target_norm=target_norm,
        relative_residual=float(relative_residual),
        rmse=rmse,
        max_abs_residual=float(np.max(np.abs(residual))) if residual.size else 0.0,
        mean_residual=float(np.mean(residual)) if residual.size else 0.0,
    )


def evaluate_pointwise_equation_residual(
    result: object,
    coefficients: Sequence[float] | np.ndarray | Mapping[str, float],
) -> PointwiseResidualEvaluationResult:
    """Evaluate residuals for an existing pointwise NS or Q fit result.

    The input ``result`` must expose the attributes
    ``pointwise_target``, ``pointwise_features``, and ``feature_names``.
    This matches the repository's current pointwise velocity and Q-equation
    result dataclasses.
    """

    try:
        target = result.pointwise_target
        features = result.pointwise_features
        feature_names = result.feature_names
    except AttributeError as exc:
        raise TypeError(
            "result must provide pointwise_target, pointwise_features, and feature_names."
        ) from exc

    return evaluate_feature_model_residual(
        target=target,
        features=features,
        feature_names=feature_names,
        coefficients=coefficients,
    )
