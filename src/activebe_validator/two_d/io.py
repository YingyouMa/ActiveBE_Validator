"""High-level NumPy workflows for pure two-dimensional validator inputs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .q_terms import reconstruct_q_tensor
from .validation import fit_be_q_equation_pointwise


def discover_q_from_processed_npy(
    q_npy_path: str | Path,
    velocity_npy_path: str | Path,
    *,
    dt: float = 1.0,
    spacings: Sequence[float] = (1.0, 1.0),
    gamma: float = 1.0,
    time_discretization: str = "two_point",
    rhs_time_level: str = "current",
    sample_step: int | Sequence[int] = 1,
    threshold: float = 1e-10,
    regularization_strength: float = 0.0,
    normalize_columns: bool = False,
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    result_output_path: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    """Recover pure-2D Q-equation parameters from processed NumPy stacks.

    The Q input must have shape ``(T, Nx, Ny, 2)`` storing ``(Qxx, Qxy)``.
    Velocity must have shape ``(T, Nx, Ny, 2)`` storing ``(ux, uy)``.
    """
    if gamma == 0.0:
        raise ValueError("gamma must be nonzero to infer free-energy parameters.")

    q_components = np.load(Path(q_npy_path))
    velocity_field = np.load(Path(velocity_npy_path))
    if q_components.ndim != 4 or q_components.shape[-1] != 2:
        raise ValueError(
            "Pure-2D Q input must have shape (T, Nx, Ny, 2); "
            f"got {q_components.shape}."
        )
    if velocity_field.ndim != 4 or velocity_field.shape[-1] != 2:
        raise ValueError(
            "Pure-2D velocity input must have shape (T, Nx, Ny, 2); "
            f"got {velocity_field.shape}."
        )
    if q_components.shape[:-1] != velocity_field.shape[:-1]:
        raise ValueError("Q and velocity stacks must share the same (T, Nx, Ny) layout.")

    q_tensor = reconstruct_q_tensor(q_components)
    fit = fit_be_q_equation_pointwise(
        velocity_field,
        q_tensor,
        dt=dt,
        spacings=spacings,
        time_axis=0,
        spatial_axes=(1, 2),
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
        time_discretization=time_discretization,
        rhs_time_level=rhs_time_level,
        sample_step=sample_step,
        threshold=threshold,
        optimizer_alpha=regularization_strength,
        normalize_columns=normalize_columns,
    )

    result: dict[str, dict[str, object]] = {
        "coefficients": {
            "lambda_c": fit.material_q_coefficient,
            "lambda_r": fit.omega_q_coefficient,
            "lambda_1": fit.e_plain_coefficient,
            "lambda_3": fit.q_colon_e_q_coefficient,
            "a2": -fit.bulk_linear_coefficient / gamma,
            "a4": -fit.bulk_quartic_coefficient / gamma,
            "L1": fit.elastic_l1_coefficient / gamma,
        },
        "metrics": {
            "r2": fit.r2,
            "rmse": fit.rmse,
            "relative_residual": fit.relative_residual,
        },
        "input_parameters": {
            "q_npy_path": str(Path(q_npy_path)),
            "velocity_npy_path": str(Path(velocity_npy_path)),
            "dt": float(dt),
            "spacings": tuple(float(value) for value in spacings),
            "gamma": float(gamma),
            "time_discretization": time_discretization,
            "rhs_time_level": rhs_time_level,
            "periodic": bool(periodic),
            "spatial_derivative_method": spatial_derivative_method,
        },
    }
    if result_output_path is not None:
        output_path = Path(result_output_path)
        if output_path.suffix.lower() != ".json":
            output_path = output_path / "q_discovery_result_2d.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["discover_q_from_processed_npy"]
