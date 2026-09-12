from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import linalg

from .ns_checker import fit_be_equation_velocity_local_weak_form
from .q_checker import BEQEquationPointwiseFitResult, fit_be_q_equation_pointwise


@dataclass(frozen=True)
class PressureLeftVelocityDiscoveryResult:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    coefficient_by_name: dict[str, float]
    prediction: np.ndarray
    target: np.ndarray
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int


@dataclass(frozen=True)
class InferredQMaterialParameters:
    gamma: float
    lambda_r: float
    lambda_1: float
    lambda_2: float
    lambda_3: float
    a2: float
    a3: float
    a4: float
    kappa: float


@dataclass(frozen=True)
class ProcessedOutputDiscoveryResult:
    inferred_q_parameters: InferredQMaterialParameters
    q_result: BEQEquationPointwiseFitResult
    velocity_result: PressureLeftVelocityDiscoveryResult


@dataclass(frozen=True)
class ProcessedQDiscoveryResult:
    inferred_q_parameters: InferredQMaterialParameters
    q_result: BEQEquationPointwiseFitResult


def _reconstruct_q_tensor(q_components: np.ndarray) -> np.ndarray:
    q_components = np.asarray(q_components, dtype=float)
    if q_components.ndim != 5 or q_components.shape[-1] != 5:
        raise ValueError(
            "Q components must have shape (T, Nx, Ny, Nz, 5); got "
            f"{q_components.shape}."
        )
    qxx = q_components[..., 0]
    qxy = q_components[..., 1]
    qxz = q_components[..., 2]
    qyy = q_components[..., 3]
    qyz = q_components[..., 4]
    qzz = -(qxx + qyy)

    q_tensor = np.empty(q_components.shape[:-1] + (3, 3), dtype=float)
    q_tensor[..., 0, 0] = qxx
    q_tensor[..., 0, 1] = qxy
    q_tensor[..., 0, 2] = qxz
    q_tensor[..., 1, 0] = qxy
    q_tensor[..., 1, 1] = qyy
    q_tensor[..., 1, 2] = qyz
    q_tensor[..., 2, 0] = qxz
    q_tensor[..., 2, 1] = qyz
    q_tensor[..., 2, 2] = qzz
    return q_tensor


def _build_masks(
    nx: int,
    ny: int,
    nz: int,
    *,
    interior_strip: int,
    boundary_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    xi = np.zeros(nx, dtype=bool)
    yi = np.zeros(ny, dtype=bool)
    zi = np.zeros(nz, dtype=bool)
    xi[interior_strip : nx - interior_strip] = True
    yi[interior_strip : ny - interior_strip] = True
    zi[interior_strip : nz - interior_strip] = True
    interior = xi[:, None, None] & yi[None, :, None] & zi[None, None, :]

    xb = np.zeros(nx, dtype=bool)
    yb = np.zeros(ny, dtype=bool)
    zb = np.zeros(nz, dtype=bool)
    xb[:boundary_layers] = True
    xb[nx - boundary_layers :] = True
    yb[:boundary_layers] = True
    yb[ny - boundary_layers :] = True
    zb[:boundary_layers] = True
    zb[nz - boundary_layers :] = True
    boundary = xb[:, None, None] | yb[None, :, None] | zb[None, None, :]
    return interior, boundary


def _sampled_point_indices(
    mask: np.ndarray,
    n_time_windows: int,
    n_sample: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, zs = np.where(mask)
    if xs.size == 0:
        raise ValueError("Sampling mask selected no spatial points.")
    t_idx = rng.integers(0, n_time_windows, size=n_sample)
    sp_idx = rng.integers(0, xs.size, size=n_sample)
    return t_idx, xs[sp_idx], ys[sp_idx], zs[sp_idx]


def _interior_q_sample_slices(
    spatial_shape: tuple[int, int, int],
    *,
    interior_strip: int,
    sample_step: int | Sequence[int],
) -> tuple[slice, slice, slice]:
    if isinstance(sample_step, (int, np.integer)):
        sample_steps = (int(sample_step),) * 3
    else:
        sample_steps = tuple(int(step) for step in sample_step)
        if len(sample_steps) != 3:
            raise ValueError("q_sample_step must be an int or a length-3 sequence.")

    if any(step <= 0 for step in sample_steps):
        raise ValueError("q_sample_step entries must be positive integers.")
    if interior_strip < 0:
        raise ValueError("interior_strip must be nonnegative.")

    sample_slices: list[slice] = []
    for axis_length, step in zip(spatial_shape, sample_steps):
        start = interior_strip
        stop = axis_length - interior_strip
        if stop <= start:
            raise ValueError(
                "interior_strip leaves no interior Q samples for spatial shape "
                f"{spatial_shape}."
            )
        sample_slices.append(slice(start, stop, step))
    return tuple(sample_slices)


def _gather_tensor(
    tensor_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    dt: int = 0,
    dx: int = 0,
    dy: int = 0,
    dz: int = 0,
    periodic: bool,
) -> np.ndarray:
    t_count, nx, ny, nz = tensor_field.shape[:4]
    ti = np.clip(t + dt, 0, t_count - 1)
    if periodic:
        return tensor_field[
            ti,
            (x + dx) % nx,
            (y + dy) % ny,
            (z + dz) % nz,
        ]
    return tensor_field[
        ti,
        np.clip(x + dx, 0, nx - 1),
        np.clip(y + dy, 0, ny - 1),
        np.clip(z + dz, 0, nz - 1),
    ]


def _gather_vector(
    vector_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    dt: int = 0,
    dx: int = 0,
    dy: int = 0,
    dz: int = 0,
    periodic: bool,
) -> np.ndarray:
    t_count, nx, ny, nz = vector_field.shape[:4]
    ti = np.clip(t + dt, 0, t_count - 1)
    if periodic:
        return vector_field[
            ti,
            (x + dx) % nx,
            (y + dy) % ny,
            (z + dz) % nz,
        ]
    return vector_field[
        ti,
        np.clip(x + dx, 0, nx - 1),
        np.clip(y + dy, 0, ny - 1),
        np.clip(z + dz, 0, nz - 1),
    ]


def _gather_scalar(
    scalar_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    dt: int = 0,
    dx: int = 0,
    dy: int = 0,
    dz: int = 0,
    periodic: bool,
) -> np.ndarray:
    t_count, nx, ny, nz = scalar_field.shape[:4]
    ti = np.clip(t + dt, 0, t_count - 1)
    if periodic:
        return scalar_field[
            ti,
            (x + dx) % nx,
            (y + dy) % ny,
            (z + dz) % nz,
        ]
    return scalar_field[
        ti,
        np.clip(x + dx, 0, nx - 1),
        np.clip(y + dy, 0, ny - 1),
        np.clip(z + dz, 0, nz - 1),
    ]


def _laplacian_u_pts(
    velocity_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    spacings: tuple[float, float, float],
    periodic: bool,
) -> np.ndarray:
    u = _gather_vector(velocity_field, t, x, y, z, periodic=periodic)
    return (
        (
            _gather_vector(velocity_field, t, x, y, z, dx=1, periodic=periodic)
            - 2.0 * u
            + _gather_vector(velocity_field, t, x, y, z, dx=-1, periodic=periodic)
        )
        / (spacings[0] ** 2)
        + (
            _gather_vector(velocity_field, t, x, y, z, dy=1, periodic=periodic)
            - 2.0 * u
            + _gather_vector(velocity_field, t, x, y, z, dy=-1, periodic=periodic)
        )
        / (spacings[1] ** 2)
        + (
            _gather_vector(velocity_field, t, x, y, z, dz=1, periodic=periodic)
            - 2.0 * u
            + _gather_vector(velocity_field, t, x, y, z, dz=-1, periodic=periodic)
        )
        / (spacings[2] ** 2)
    )


def grad_div_u_pts(
    velocity_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    spacings: tuple[float, float, float],
    periodic: bool,
) -> np.ndarray:
    """Return grad(div(u)) sampled at points using centered differences."""

    def divergence_at(dx: int, dy: int, dz: int) -> np.ndarray:
        duxdx = (
            _gather_vector(
                velocity_field,
                t,
                x,
                y,
                z,
                dx=dx + 1,
                dy=dy,
                dz=dz,
                periodic=periodic,
            )[..., 0]
            - _gather_vector(
                velocity_field,
                t,
                x,
                y,
                z,
                dx=dx - 1,
                dy=dy,
                dz=dz,
                periodic=periodic,
            )[..., 0]
        ) / (2.0 * spacings[0])
        duydy = (
            _gather_vector(
                velocity_field,
                t,
                x,
                y,
                z,
                dx=dx,
                dy=dy + 1,
                dz=dz,
                periodic=periodic,
            )[..., 1]
            - _gather_vector(
                velocity_field,
                t,
                x,
                y,
                z,
                dx=dx,
                dy=dy - 1,
                dz=dz,
                periodic=periodic,
            )[..., 1]
        ) / (2.0 * spacings[1])
        duzdz = (
            _gather_vector(
                velocity_field,
                t,
                x,
                y,
                z,
                dx=dx,
                dy=dy,
                dz=dz + 1,
                periodic=periodic,
            )[..., 2]
            - _gather_vector(
                velocity_field,
                t,
                x,
                y,
                z,
                dx=dx,
                dy=dy,
                dz=dz - 1,
                periodic=periodic,
            )[..., 2]
        ) / (2.0 * spacings[2])
        return duxdx + duydy + duzdz

    return np.stack(
        [
            (divergence_at(1, 0, 0) - divergence_at(-1, 0, 0))
            / (2.0 * spacings[0]),
            (divergence_at(0, 1, 0) - divergence_at(0, -1, 0))
            / (2.0 * spacings[1]),
            (divergence_at(0, 0, 1) - divergence_at(0, 0, -1))
            / (2.0 * spacings[2]),
        ],
        axis=-1,
    )


def _grad_u_tensor_pts(
    velocity_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    spacings: tuple[float, float, float],
    periodic: bool,
) -> np.ndarray:
    gx = (
        _gather_vector(velocity_field, t, x, y, z, dx=1, periodic=periodic)
        - _gather_vector(velocity_field, t, x, y, z, dx=-1, periodic=periodic)
    ) / (2.0 * spacings[0])
    gy = (
        _gather_vector(velocity_field, t, x, y, z, dy=1, periodic=periodic)
        - _gather_vector(velocity_field, t, x, y, z, dy=-1, periodic=periodic)
    ) / (2.0 * spacings[1])
    gz = (
        _gather_vector(velocity_field, t, x, y, z, dz=1, periodic=periodic)
        - _gather_vector(velocity_field, t, x, y, z, dz=-1, periodic=periodic)
    ) / (2.0 * spacings[2])
    return np.stack([gx, gy, gz], axis=-2)


def _laplacian_q_pts(
    q_tensor_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    spacings: tuple[float, float, float],
    periodic: bool,
) -> np.ndarray:
    q = _gather_tensor(q_tensor_field, t, x, y, z, periodic=periodic)
    return (
        (
            _gather_tensor(q_tensor_field, t, x, y, z, dx=1, periodic=periodic)
            - 2.0 * q
            + _gather_tensor(q_tensor_field, t, x, y, z, dx=-1, periodic=periodic)
        )
        / (spacings[0] ** 2)
        + (
            _gather_tensor(q_tensor_field, t, x, y, z, dy=1, periodic=periodic)
            - 2.0 * q
            + _gather_tensor(q_tensor_field, t, x, y, z, dy=-1, periodic=periodic)
        )
        / (spacings[1] ** 2)
        + (
            _gather_tensor(q_tensor_field, t, x, y, z, dz=1, periodic=periodic)
            - 2.0 * q
            + _gather_tensor(q_tensor_field, t, x, y, z, dz=-1, periodic=periodic)
        )
        / (spacings[2] ** 2)
    )


def _grad_q_pts(
    q_tensor_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    spacings: tuple[float, float, float],
    periodic: bool,
) -> np.ndarray:
    gx = (
        _gather_tensor(q_tensor_field, t, x, y, z, dx=1, periodic=periodic)
        - _gather_tensor(q_tensor_field, t, x, y, z, dx=-1, periodic=periodic)
    ) / (2.0 * spacings[0])
    gy = (
        _gather_tensor(q_tensor_field, t, x, y, z, dy=1, periodic=periodic)
        - _gather_tensor(q_tensor_field, t, x, y, z, dy=-1, periodic=periodic)
    ) / (2.0 * spacings[1])
    gz = (
        _gather_tensor(q_tensor_field, t, x, y, z, dz=1, periodic=periodic)
        - _gather_tensor(q_tensor_field, t, x, y, z, dz=-1, periodic=periodic)
    ) / (2.0 * spacings[2])
    return np.stack([gx, gy, gz], axis=-3)


def _div_q_pts(
    q_tensor_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    spacings: tuple[float, float, float],
    periodic: bool,
) -> np.ndarray:
    dqdx = (
        _gather_tensor(q_tensor_field, t, x, y, z, dx=1, periodic=periodic)
        - _gather_tensor(q_tensor_field, t, x, y, z, dx=-1, periodic=periodic)
    ) / (2.0 * spacings[0])
    dqdy = (
        _gather_tensor(q_tensor_field, t, x, y, z, dy=1, periodic=periodic)
        - _gather_tensor(q_tensor_field, t, x, y, z, dy=-1, periodic=periodic)
    ) / (2.0 * spacings[1])
    dqdz = (
        _gather_tensor(q_tensor_field, t, x, y, z, dz=1, periodic=periodic)
        - _gather_tensor(q_tensor_field, t, x, y, z, dz=-1, periodic=periodic)
    ) / (2.0 * spacings[2])
    return dqdx[..., :, 0] + dqdy[..., :, 1] + dqdz[..., :, 2]


def _sigma_pts(
    q: np.ndarray,
    lapq: np.ndarray,
    gradq: np.ndarray,
    *,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
    flow_alignment_xi: float,
) -> np.ndarray:
    identity = np.eye(3)
    trace_q2 = np.einsum("...ij,...ij->...", q, q)
    q_squared = np.einsum("...ik,...kj->...ij", q, q)
    trace_q3 = np.einsum("...ij,...ji->...", q_squared, q)
    grad_norm_sq = np.einsum("...dij,...dij->...", gradq, gradq)
    free_energy_density = (
        0.5 * a2 * trace_q2
        + (a3 / 3.0) * trace_q3
        + 0.25 * a4 * trace_q2**2
        + 0.5 * kappa * grad_norm_sq
    )

    molecular_field = (
        -a2 * q
        - a3 * (q_squared - trace_q2[..., np.newaxis, np.newaxis] * identity / 3.0)
        - a4 * trace_q2[..., np.newaxis, np.newaxis] * q
        + kappa * lapq
    )
    shifted_q = q + identity / 3.0
    q_h_scalar = np.einsum("...ij,...ij->...", q, molecular_field)

    sigma = -free_energy_density[..., np.newaxis, np.newaxis] * identity
    sigma -= (
        2.0
        * flow_alignment_xi
        * shifted_q
        * q_h_scalar[..., np.newaxis, np.newaxis]
    )
    sigma += flow_alignment_xi * np.einsum(
        "...ik,...jk->...ij", molecular_field, shifted_q
    )
    sigma += flow_alignment_xi * np.einsum(
        "...ik,...jk->...ij", shifted_q, molecular_field
    )
    sigma += np.einsum("...ik,...kj->...ij", molecular_field, q)
    sigma -= np.einsum("...ik,...kj->...ij", q, molecular_field)
    sigma += kappa * np.einsum("...acd,...bcd->...ab", gradq, gradq)
    return sigma


def _molecular_field_pts(
    q: np.ndarray,
    lapq: np.ndarray,
    *,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
) -> np.ndarray:
    identity = np.eye(3)
    trace_q2 = np.einsum("...ij,...ij->...", q, q)
    q_squared = np.einsum("...ik,...kj->...ij", q, q)
    return (
        -a2 * q
        - a3 * (q_squared - trace_q2[..., np.newaxis, np.newaxis] * identity / 3.0)
        - a4 * trace_q2[..., np.newaxis, np.newaxis] * q
        + kappa * lapq
    )


def _infer_q_material_parameters(
    q_result: BEQEquationPointwiseFitResult,
    *,
    gamma: float,
) -> InferredQMaterialParameters:
    if np.isclose(gamma, 0.0):
        raise ValueError("gamma must be nonzero to infer a2, a3, a4, and kappa.")
    return InferredQMaterialParameters(
        gamma=float(gamma),
        lambda_r=float(q_result.omega_q_coefficient),
        lambda_1=float(q_result.e_plain_coefficient),
        lambda_2=float(q_result.e_q_coefficient),
        lambda_3=float(q_result.q_colon_e_q_coefficient),
        a2=float(-q_result.bulk_linear_coefficient / gamma),
        a3=float(-q_result.bulk_quadratic_coefficient / gamma),
        a4=float(-q_result.bulk_cubic_coefficient / gamma),
        kappa=float(q_result.elastic_l1_coefficient / gamma),
    )


def _discover_q_from_processed_npy(
    q_npy_path: str | Path,
    velocity_npy_path: str | Path,
    *,
    dt: float = 1.0,
    spacings: Sequence[float] = (1.0, 1.0, 1.0),
    q_time_discretization: str = "two_point",
    v_time_level: str = "n+1",
    q_time_level: str = "n",
    q_sample_step: int | Sequence[int] = 8,
    q_threshold: float = 1e-8,
    q_optimizer_alpha: float = 1e-12,
    is_q_normalize_columns: bool = True,
    interior_strip: int = 3,
    gamma: float = 1.0,
    is_verbose: bool = False,
) -> ProcessedQDiscoveryResult:
    q_components = np.load(Path(q_npy_path))
    velocity_field = np.load(Path(velocity_npy_path))

    q_tensor_field = _reconstruct_q_tensor(q_components)
    if velocity_field.ndim != 5 or velocity_field.shape[-1] != 3:
        raise ValueError(
            "Velocity stack must have shape (T, Nx, Ny, Nz, 3); got "
            f"{velocity_field.shape}."
        )
    if velocity_field.shape[:-1] != q_tensor_field.shape[:-2]:
        raise ValueError(
            "Q and velocity stacks must share the same (T, Nx, Ny, Nz) layout; "
            f"got velocity shape {velocity_field.shape} and Q shape "
            f"{q_tensor_field.shape}."
        )

    effective_periodic = False
    q_spatial_derivative_method = "upwind3"
    q_include_terms = (
        "material_q",
        "omega_q",
        "e_plain",
        "e_q",
        "q_colon_e_q",
        "bulk_linear",
        "bulk_quadratic",
        "bulk_cubic",
        "elastic_l1",
    )

    q_sample_slices = _interior_q_sample_slices(
        q_tensor_field.shape[1:4],
        interior_strip=interior_strip,
        sample_step=q_sample_step,
    )
    q_result = fit_be_q_equation_pointwise(
        velocity_field=velocity_field,
        q_tensor_field=q_tensor_field,
        dt=dt,
        spacings=tuple(float(value) for value in spacings),
        time_axis=0,
        spatial_axes=(1, 2, 3),
        periodic=effective_periodic,
        spatial_derivative_method=q_spatial_derivative_method,
        v_time_level=v_time_level,
        q_time_level=q_time_level,
        time_discretization=q_time_discretization,
        use_interior_time_only=False,
        threshold=q_threshold,
        optimizer_alpha=q_optimizer_alpha,
        normalize_columns=is_q_normalize_columns,
        include_terms=q_include_terms,
        sample_slices=q_sample_slices,
        verbose=is_verbose,
    )
    inferred_q_parameters = _infer_q_material_parameters(q_result, gamma=gamma)
    return ProcessedQDiscoveryResult(
        inferred_q_parameters=inferred_q_parameters,
        q_result=q_result,
    )


def discover_q_from_processed_npy(
    q_npy_path: str | Path,
    velocity_npy_path: str | Path,
    *,
    dt: float = 1.0,
    spacings: Sequence[float] = (1.0, 1.0, 1.0),
    time_discretization: str = "two_point",
    velocity_time_level: str = "n+1",
    q_time_level: str = "n",
    sample_step: int | Sequence[int] = 8,
    threshold: float = 1e-8,
    regularization_strength: float = 1e-12,
    is_normalize_columns: bool = True,
    interior_strip: int = 3,
    gamma: float = 1.0,
    is_verbose: bool = False,
    result_output_path: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    """Discover Q-equation parameters and return a notebook-friendly summary."""
    discovery = _discover_q_from_processed_npy(
        q_npy_path=q_npy_path,
        velocity_npy_path=velocity_npy_path,
        dt=dt,
        spacings=spacings,
        q_time_discretization=time_discretization,
        v_time_level=velocity_time_level,
        q_time_level=q_time_level,
        q_sample_step=sample_step,
        q_threshold=threshold,
        q_optimizer_alpha=regularization_strength,
        is_q_normalize_columns=is_normalize_columns,
        interior_strip=interior_strip,
        gamma=gamma,
        is_verbose=is_verbose,
    )
    parameters = discovery.inferred_q_parameters
    q_result = discovery.q_result
    sample_step = (
        int(sample_step)
        if isinstance(sample_step, (int, np.integer))
        else tuple(int(step) for step in sample_step)
    )

    result = {
        "coefficients": {
            "lambda_c": q_result.material_q_coefficient,
            "lambda_r": parameters.lambda_r,
            "lambda_1": parameters.lambda_1,
            "lambda_2": parameters.lambda_2,
            "lambda_3": parameters.lambda_3,
            "a2": parameters.a2,
            "a3": parameters.a3,
            "a4": parameters.a4,
            "L1": parameters.kappa,
        },
        "input_parameters": {
            "q_npy_path": str(Path(q_npy_path)),
            "velocity_npy_path": str(Path(velocity_npy_path)),
            "dt": float(dt),
            "spacings": tuple(float(value) for value in spacings),
            "time_discretization": time_discretization,
            "velocity_time_level": velocity_time_level,
            "q_time_level": q_time_level,
            "sample_step": sample_step,
            "threshold": float(threshold),
            "regularization_strength": float(regularization_strength),
            "is_normalize_columns": bool(is_normalize_columns),
            "interior_strip": int(interior_strip),
            "gamma": float(gamma),
            "is_verbose": bool(is_verbose),
            "result_output_path": (
                None if result_output_path is None else str(Path(result_output_path))
            ),
        },
        "metrics": {
            "r2": q_result.r2,
            "relative_residual": q_result.relative_residual,
            "rmse": q_result.rmse,
        },
    }

    if result_output_path is not None:
        output_path = Path(result_output_path)
        if output_path.suffix.lower() != ".json":
            output_path = output_path / "q_discovery_result.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)

    return result


def _passive_stress_div_pts(
    q_tensor_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    spacings: tuple[float, float, float],
    periodic: bool,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
    flow_alignment_xi: float,
) -> np.ndarray:
    def sigma_at(dx: int, dy: int, dz: int) -> np.ndarray:
        q = _gather_tensor(q_tensor_field, t, x, y, z, dx=dx, dy=dy, dz=dz, periodic=periodic)
        qxp = _gather_tensor(q_tensor_field, t, x, y, z, dx=dx + 1, dy=dy, dz=dz, periodic=periodic)
        qxm = _gather_tensor(q_tensor_field, t, x, y, z, dx=dx - 1, dy=dy, dz=dz, periodic=periodic)
        qyp = _gather_tensor(q_tensor_field, t, x, y, z, dx=dx, dy=dy + 1, dz=dz, periodic=periodic)
        qym = _gather_tensor(q_tensor_field, t, x, y, z, dx=dx, dy=dy - 1, dz=dz, periodic=periodic)
        qzp = _gather_tensor(q_tensor_field, t, x, y, z, dx=dx, dy=dy, dz=dz + 1, periodic=periodic)
        qzm = _gather_tensor(q_tensor_field, t, x, y, z, dx=dx, dy=dy, dz=dz - 1, periodic=periodic)

        lapq = (
            (qxp - 2.0 * q + qxm) / (spacings[0] ** 2)
            + (qyp - 2.0 * q + qym) / (spacings[1] ** 2)
            + (qzp - 2.0 * q + qzm) / (spacings[2] ** 2)
        )
        gradq = np.stack(
            [
                (qxp - qxm) / (2.0 * spacings[0]),
                (qyp - qym) / (2.0 * spacings[1]),
                (qzp - qzm) / (2.0 * spacings[2]),
            ],
            axis=-3,
        )
        return _sigma_pts(
            q,
            lapq,
            gradq,
            a2=a2,
            a3=a3,
            a4=a4,
            kappa=kappa,
            flow_alignment_xi=flow_alignment_xi,
        )

    sig_xp = sigma_at(1, 0, 0)
    sig_xm = sigma_at(-1, 0, 0)
    sig_yp = sigma_at(0, 1, 0)
    sig_ym = sigma_at(0, -1, 0)
    sig_zp = sigma_at(0, 0, 1)
    sig_zm = sigma_at(0, 0, -1)

    dsig_dx = (sig_xp - sig_xm) / (2.0 * spacings[0])
    dsig_dy = (sig_yp - sig_ym) / (2.0 * spacings[1])
    dsig_dz = (sig_zp - sig_zm) / (2.0 * spacings[2])
    return dsig_dx[..., :, 0] + dsig_dy[..., :, 1] + dsig_dz[..., :, 2]


def _advective_backflow_pts(
    q_tensor_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    spacings: tuple[float, float, float],
    periodic: bool,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
) -> np.ndarray:
    q = _gather_tensor(q_tensor_field, t, x, y, z, periodic=periodic)
    lapq = _laplacian_q_pts(
        q_tensor_field,
        t,
        x,
        y,
        z,
        spacings=spacings,
        periodic=periodic,
    )
    gradq = _grad_q_pts(
        q_tensor_field,
        t,
        x,
        y,
        z,
        spacings=spacings,
        periodic=periodic,
    )
    molecular_field = _molecular_field_pts(
        q,
        lapq,
        a2=a2,
        a3=a3,
        a4=a4,
        kappa=kappa,
    )

    qxxx = gradq[..., 0, 0, 0]
    qxyx = gradq[..., 0, 0, 1]
    qxzx = gradq[..., 0, 0, 2]
    qyyx = gradq[..., 0, 1, 1]
    qyzx = gradq[..., 0, 1, 2]
    qxxy = gradq[..., 1, 0, 0]
    qxyy = gradq[..., 1, 0, 1]
    qxzy = gradq[..., 1, 0, 2]
    qyyy = gradq[..., 1, 1, 1]
    qyzy = gradq[..., 1, 1, 2]
    qxxz = gradq[..., 2, 0, 0]
    qxyz = gradq[..., 2, 0, 1]
    qxzz = gradq[..., 2, 0, 2]
    qyyz = gradq[..., 2, 1, 1]
    qyzz = gradq[..., 2, 1, 2]

    hxx = molecular_field[..., 0, 0]
    hxy = molecular_field[..., 0, 1]
    hxz = molecular_field[..., 0, 2]
    hyy = molecular_field[..., 1, 1]
    hyz = molecular_field[..., 1, 2]

    return np.stack(
        [
            -2.0 * (hxx * qxxx + hxy * qxyx + hxz * qxzx + hyy * qyyx + hyz * qyzx)
            + hxx * qyyx
            + hyy * qxxx,
            -2.0 * (hxx * qxxy + hxy * qxyy + hxz * qxzy + hyy * qyyy + hyz * qyzy)
            + hxx * qyyy
            + hyy * qxxy,
            -2.0 * (hxx * qxxz + hxy * qxyz + hxz * qxzz + hyy * qyyz + hyz * qyzz)
            + hxx * qyyz
            + hyy * qxxz,
        ],
        axis=-1,
    )


def _lbman3d_passive_stress_pts(
    q: np.ndarray,
    molecular_field: np.ndarray,
    *,
    flow_alignment_xi: float,
) -> np.ndarray:
    identity = np.eye(3)
    hq_plus_qh = np.einsum("...ik,...kj->...ij", molecular_field, q) + np.einsum(
        "...ik,...kj->...ij", q, molecular_field
    )
    qh_symtr = 0.5 * (hq_plus_qh + np.swapaxes(hq_plus_qh, -1, -2))
    qh_symtr -= (
        np.einsum("...ii->...", qh_symtr)[..., np.newaxis, np.newaxis] * identity / 3.0
    )
    tau = np.einsum("...ik,...kj->...ij", molecular_field, q) - np.einsum(
        "...ik,...kj->...ij", q, molecular_field
    )
    return (
        -(2.0 / 3.0) * flow_alignment_xi * molecular_field
        - flow_alignment_xi * qh_symtr
        + tau
    )


def lbman3d_passive_backflow_pts(
    q_tensor_field: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    spacings: tuple[float, float, float],
    periodic: bool,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
    flow_alignment_xi: float,
) -> np.ndarray:
    """Return the `lbman3d`-specific passive backflow force `div(P)` at points.

    This matches the passive stress actually written into `qf.Pxx/Pxy/Pxz/Pyy/Pyz`
    in `lbman3d/src/model.h`, rather than the more complete generic passive stress
    basis used elsewhere in this repository.
    """

    def stress_at(dx: int, dy: int, dz: int) -> np.ndarray:
        q = _gather_tensor(
            q_tensor_field,
            t,
            x,
            y,
            z,
            dx=dx,
            dy=dy,
            dz=dz,
            periodic=periodic,
        )
        qxp = _gather_tensor(
            q_tensor_field,
            t,
            x,
            y,
            z,
            dx=dx + 1,
            dy=dy,
            dz=dz,
            periodic=periodic,
        )
        qxm = _gather_tensor(
            q_tensor_field,
            t,
            x,
            y,
            z,
            dx=dx - 1,
            dy=dy,
            dz=dz,
            periodic=periodic,
        )
        qyp = _gather_tensor(
            q_tensor_field,
            t,
            x,
            y,
            z,
            dx=dx,
            dy=dy + 1,
            dz=dz,
            periodic=periodic,
        )
        qym = _gather_tensor(
            q_tensor_field,
            t,
            x,
            y,
            z,
            dx=dx,
            dy=dy - 1,
            dz=dz,
            periodic=periodic,
        )
        qzp = _gather_tensor(
            q_tensor_field,
            t,
            x,
            y,
            z,
            dx=dx,
            dy=dy,
            dz=dz + 1,
            periodic=periodic,
        )
        qzm = _gather_tensor(
            q_tensor_field,
            t,
            x,
            y,
            z,
            dx=dx,
            dy=dy,
            dz=dz - 1,
            periodic=periodic,
        )

        lapq = (
            (qxp - 2.0 * q + qxm) / (spacings[0] ** 2)
            + (qyp - 2.0 * q + qym) / (spacings[1] ** 2)
            + (qzp - 2.0 * q + qzm) / (spacings[2] ** 2)
        )
        molecular_field = _molecular_field_pts(
            q,
            lapq,
            a2=a2,
            a3=a3,
            a4=a4,
            kappa=kappa,
        )
        return _lbman3d_passive_stress_pts(
            q,
            molecular_field,
            flow_alignment_xi=flow_alignment_xi,
        )

    sig_xp = stress_at(1, 0, 0)
    sig_xm = stress_at(-1, 0, 0)
    sig_yp = stress_at(0, 1, 0)
    sig_ym = stress_at(0, -1, 0)
    sig_zp = stress_at(0, 0, 1)
    sig_zm = stress_at(0, 0, -1)

    dsig_dx = (sig_xp - sig_xm) / (2.0 * spacings[0])
    dsig_dy = (sig_yp - sig_ym) / (2.0 * spacings[1])
    dsig_dz = (sig_zp - sig_zm) / (2.0 * spacings[2])
    return dsig_dx[..., :, 0] + dsig_dy[..., :, 1] + dsig_dz[..., :, 2]


def discover_from_processed_npy(
    q_npy_path: str | Path,
    velocity_npy_path: str | Path,
    density_npy_path: str | Path,
    *,
    dt: float = 1.0,
    spacings: Sequence[float] = (1.0, 1.0, 1.0),
    q_time_discretization: str = "two_point",
    v_time_level: str = "n+1",
    q_time_level: str = "n",
    q_sample_step: int | Sequence[int] = 8,
    q_threshold: float = 1e-8,
    q_optimizer_alpha: float = 1e-12,
    is_q_normalize_columns: bool = True,
    velocity_n_sample: int = 5000,
    velocity_rng_seed: int = 1234,
    interior_strip: int = 3,
    boundary_layers: int = 2,
    cs2: float = 1.0 / 3.0,
    gamma: float = 1.0,
    is_verbose: bool = False,
) -> ProcessedOutputDiscoveryResult:
    q_components = np.load(Path(q_npy_path))
    velocity_field = np.load(Path(velocity_npy_path))
    density_field = np.load(Path(density_npy_path))

    q_tensor_field = _reconstruct_q_tensor(q_components)
    if velocity_field.ndim != 5 or velocity_field.shape[-1] != 3:
        raise ValueError(
            "Velocity stack must have shape (T, Nx, Ny, Nz, 3); got "
            f"{velocity_field.shape}."
        )
    if density_field.ndim != 4:
        raise ValueError(
            "Density stack must have shape (T, Nx, Ny, Nz); got "
            f"{density_field.shape}."
        )
    if velocity_field.shape[:-1] != density_field.shape or velocity_field.shape[:-1] != q_tensor_field.shape[:-2]:
        raise ValueError(
            "Q, velocity, and density stacks must share the same (T, Nx, Ny, Nz) layout."
        )

    q_discovery = _discover_q_from_processed_npy(
        q_npy_path=q_npy_path,
        velocity_npy_path=velocity_npy_path,
        dt=dt,
        spacings=spacings,
        q_time_discretization=q_time_discretization,
        v_time_level=v_time_level,
        q_time_level=q_time_level,
        q_sample_step=q_sample_step,
        q_threshold=q_threshold,
        q_optimizer_alpha=q_optimizer_alpha,
        is_q_normalize_columns=is_q_normalize_columns,
        interior_strip=interior_strip,
        gamma=gamma,
        is_verbose=is_verbose,
    )
    q_result = q_discovery.q_result
    inferred_q_parameters = q_discovery.inferred_q_parameters

    _, nx, ny, nz = density_field.shape
    interior_mask, _ = _build_masks(
        nx,
        ny,
        nz,
        interior_strip=interior_strip,
        boundary_layers=boundary_layers,
    )
    rng = np.random.default_rng(velocity_rng_seed)
    t_idx, x_idx, y_idx, z_idx = _sampled_point_indices(
        interior_mask,
        velocity_field.shape[0] - 1,
        velocity_n_sample,
        rng,
    )

    spacings_tuple = tuple(float(value) for value in spacings)
    u_t = _gather_vector(
        velocity_field, t_idx, x_idx, y_idx, z_idx, periodic=False
    )
    u_tp1 = _gather_vector(
        velocity_field, t_idx, x_idx, y_idx, z_idx, dt=1, periodic=False
    )
    dt_v = (u_tp1 - u_t) / dt

    viscous_v = _laplacian_u_pts(
        velocity_field,
        t_idx,
        x_idx,
        y_idx,
        z_idx,
        spacings=spacings_tuple,
        periodic=False,
    )
    active_v = _div_q_pts(
        q_tensor_field,
        t_idx,
        x_idx,
        y_idx,
        z_idx,
        spacings=spacings_tuple,
        periodic=False,
    )
    backflow_v = -_passive_stress_div_pts(
        q_tensor_field,
        t_idx,
        x_idx,
        y_idx,
        z_idx,
        spacings=spacings_tuple,
        periodic=False,
        a2=inferred_q_parameters.a2,
        a3=inferred_q_parameters.a3,
        a4=inferred_q_parameters.a4,
        kappa=inferred_q_parameters.kappa,
        flow_alignment_xi=inferred_q_parameters.lambda_2,
    )
    advective_backflow_v = _advective_backflow_pts(
        q_tensor_field,
        t_idx,
        x_idx,
        y_idx,
        z_idx,
        spacings=spacings_tuple,
        periodic=False,
        a2=inferred_q_parameters.a2,
        a3=inferred_q_parameters.a3,
        a4=inferred_q_parameters.a4,
        kappa=inferred_q_parameters.kappa,
    )
    neg_dt_v = -dt_v
    grad_u = _grad_u_tensor_pts(
        velocity_field,
        t_idx,
        x_idx,
        y_idx,
        z_idx,
        spacings=spacings_tuple,
        periodic=False,
    )
    neg_adv_v = -np.einsum("...i,...ij->...j", u_t, grad_u)

    rho_c = _gather_scalar(
        density_field, t_idx, x_idx, y_idx, z_idx, periodic=False
    )
    grad_p = np.stack(
        [
            cs2
            * (
                _gather_scalar(
                    density_field,
                    t_idx,
                    x_idx,
                    y_idx,
                    z_idx,
                    dx=1,
                    periodic=False,
                )
                - _gather_scalar(
                    density_field,
                    t_idx,
                    x_idx,
                    y_idx,
                    z_idx,
                    dx=-1,
                    periodic=False,
                )
            )
            / (2.0 * spacings_tuple[0]),
            cs2
            * (
                _gather_scalar(
                    density_field,
                    t_idx,
                    x_idx,
                    y_idx,
                    z_idx,
                    dy=1,
                    periodic=False,
                )
                - _gather_scalar(
                    density_field,
                    t_idx,
                    x_idx,
                    y_idx,
                    z_idx,
                    dy=-1,
                    periodic=False,
                )
            )
            / (2.0 * spacings_tuple[1]),
            cs2
            * (
                _gather_scalar(
                    density_field,
                    t_idx,
                    x_idx,
                    y_idx,
                    z_idx,
                    dz=1,
                    periodic=False,
                )
                - _gather_scalar(
                    density_field,
                    t_idx,
                    x_idx,
                    y_idx,
                    z_idx,
                    dz=-1,
                    periodic=False,
                )
            )
            / (2.0 * spacings_tuple[2]),
        ],
        axis=-1,
    )
    pressure_grad_over_rho = grad_p / rho_c[:, np.newaxis]

    feature_names = (
        "viscous_v",
        "active_v",
        "backflow_v",
        "advective_backflow_v",
        "neg_dt_v",
        "neg_adv_v",
    )
    feature_arrays = [
        viscous_v,
        active_v,
        backflow_v,
        advective_backflow_v,
        neg_dt_v,
        neg_adv_v,
    ]
    theta = np.column_stack(
        [feature.reshape(velocity_n_sample, 3).ravel() for feature in feature_arrays]
    )
    y_vec = pressure_grad_over_rho.ravel()

    y_scale = np.sqrt(np.mean(y_vec**2))
    col_scale = np.sqrt(np.mean(theta**2, axis=0))
    col_scale[col_scale == 0.0] = 1.0
    coefficients_scaled, *_ = linalg.lstsq(theta / col_scale, y_vec / y_scale)
    coefficients = coefficients_scaled * y_scale / col_scale
    prediction = theta @ coefficients
    residual = y_vec - prediction
    rmse = float(np.sqrt(np.mean(residual**2)))
    relative_residual = float(rmse / y_scale)
    centered_y = y_vec - np.mean(y_vec)
    r2 = float(1.0 - np.sum(residual**2) / np.sum(centered_y**2))
    coefficient_by_name = dict(zip(feature_names, coefficients.astype(float)))

    velocity_result = PressureLeftVelocityDiscoveryResult(
        feature_names=feature_names,
        coefficients=coefficients,
        coefficient_by_name=coefficient_by_name,
        prediction=prediction,
        target=y_vec,
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(velocity_n_sample * 3),
    )
    return ProcessedOutputDiscoveryResult(
        inferred_q_parameters=inferred_q_parameters,
        q_result=q_result,
        velocity_result=velocity_result,
    )


def discover_from_processed_npy_local_weak_form(
    q_npy_path: str | Path,
    velocity_npy_path: str | Path,
    *,
    dt: float = 1.0,
    spacings: Sequence[float] = (1.0, 1.0, 1.0),
    viscosity: float = 1.0,
    v_patch_half_widths: tuple[int, int, int] = (4, 4, 4),
    v_patch_strides: tuple[int, int, int] = (16, 16, 16),
    v_time_derivative_method: str = "central",
    num_test_fields_per_patch: int = 1,
    is_v_include_passive_backflow: bool = False,
    v_a2: float | None = None,
    v_a3: float | None = None,
    v_a4: float | None = None,
    v_kappa: float | None = None,
    v_flow_alignment_xi: float = 1.0,
    v_threshold: float = 1e-10,
    v_optimizer_alpha: float = 0.0,
    is_v_normalize_columns: bool = False,
    is_verbose: bool = False,
    progress_interval: int = 10,
) -> dict[str, dict[str, object]]:
    """Run local weak-form velocity discovery from processed ``.npy`` stacks.

    This entry point is solver-agnostic at the file-format level: it expects
    Q components stored as ``(T, Nx, Ny, Nz, 5)`` and velocity stored as
    ``(T, Nx, Ny, Nz, 3)``. The velocity fit uses the repository's existing
    BE-style local weak form, so pressure is eliminated by construction via
    divergence-free test fields.
    """

    q_components = np.load(Path(q_npy_path))
    velocity_field = np.load(Path(velocity_npy_path))
    q_tensor_field = _reconstruct_q_tensor(q_components)

    if velocity_field.ndim != 5 or velocity_field.shape[-1] != 3:
        raise ValueError(
            "Velocity stack must have shape (T, Nx, Ny, Nz, 3); got "
            f"{velocity_field.shape}."
        )
    if velocity_field.shape[:-1] != q_tensor_field.shape[:-2]:
        raise ValueError(
            "Q and velocity stacks must share the same (T, Nx, Ny, Nz) layout; "
            f"got velocity shape {velocity_field.shape} and Q shape "
            f"{q_tensor_field.shape}."
        )

    v_spatial_derivative_method = "finite_difference"
    v_include_distortion_stress = True

    velocity_result = fit_be_equation_velocity_local_weak_form(
        velocity_field=velocity_field,
        q_tensor_field=q_tensor_field,
        dt=dt,
        spacings=spacings,
        viscosity=viscosity,
        patch_half_widths=v_patch_half_widths,
        patch_strides=v_patch_strides,
        periodic=False,
        spatial_derivative_method=v_spatial_derivative_method,
        time_derivative_method=v_time_derivative_method,
        num_test_fields_per_patch=num_test_fields_per_patch,
        include_passive_backflow=is_v_include_passive_backflow,
        a2=v_a2,
        a3=v_a3,
        a4=v_a4,
        kappa=v_kappa,
        flow_alignment_xi=v_flow_alignment_xi,
        include_distortion_stress=v_include_distortion_stress,
        threshold=v_threshold,
        optimizer_alpha=v_optimizer_alpha,
        normalize_columns=is_v_normalize_columns,
        verbose=is_verbose,
        progress_interval=progress_interval,
    )

    return {
        "coefficients": {
            "partial t": velocity_result.dt_coefficient,
            "material_advection": velocity_result.material_coefficient,
            "viscosity": velocity_result.viscosity_coefficient,
            "active_force": velocity_result.q_divergence_coefficient,
            "passive_backflow": velocity_result.passive_backflow_coefficient,
            "linear_velocity": velocity_result.linear_velocity_coefficient,
            "quadratic_velocity": velocity_result.quadratic_velocity_coefficient,
        },
        "input_parameters": {
            "q_npy_path": str(Path(q_npy_path)),
            "velocity_npy_path": str(Path(velocity_npy_path)),
            "dt": float(dt),
            "spacings": tuple(float(value) for value in spacings),
            "viscosity": float(viscosity),
            "v_patch_half_widths": tuple(int(value) for value in v_patch_half_widths),
            "v_patch_strides": tuple(int(value) for value in v_patch_strides),
            "v_time_derivative_method": v_time_derivative_method,
            "num_test_fields_per_patch": int(num_test_fields_per_patch),
            "is_v_include_passive_backflow": bool(is_v_include_passive_backflow),
            "v_a2": None if v_a2 is None else float(v_a2),
            "v_a3": None if v_a3 is None else float(v_a3),
            "v_a4": None if v_a4 is None else float(v_a4),
            "v_kappa": None if v_kappa is None else float(v_kappa),
            "v_flow_alignment_xi": float(v_flow_alignment_xi),
            "v_threshold": float(v_threshold),
            "v_optimizer_alpha": float(v_optimizer_alpha),
            "is_v_normalize_columns": bool(is_v_normalize_columns),
            "is_verbose": bool(is_verbose),
            "progress_interval": int(progress_interval),
        },
        "metrics": {
            "r2": velocity_result.r2,
            "relative_residual": velocity_result.relative_residual,
            "rmse": velocity_result.rmse,
            "sample_count": velocity_result.sample_count,
        },
    }
