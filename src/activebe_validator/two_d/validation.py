"""Pointwise and local weak-form validators for pure two-dimensional fields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .._regression import fit_linear_term_model
from ._differential import gradient
from .q_terms import (
    bulk_linear_term,
    bulk_quartic_term,
    elastic_molecular_field_l1,
    extract_independent_q_components,
    q_colon_e_q_interaction,
    q_e_interaction,
    q_e_plain_interaction,
    q_flow_alignment,
    q_material_derivative,
    q_omega_interaction,
    q_time_derivative,
)
from .velocity_terms import (
    active_q_divergence_force,
    aligned_quadratic_velocity,
    passive_backflow_force,
    velocity_laplacian,
    velocity_material_derivative,
    velocity_shear_and_rotation_tensors,
    velocity_time_derivative_2d,
)


@dataclass(frozen=True)
class LocalPatchSpec2D:
    center_indices: tuple[int, int]
    half_widths: tuple[int, int]


@dataclass(frozen=True)
class BEQEquationPointwiseFitResult2D:
    material_q_coefficient: float
    omega_q_coefficient: float
    e_plain_coefficient: float
    e_q_coefficient: float
    q_colon_e_q_coefficient: float
    flow_alignment_coefficient: float
    bulk_linear_coefficient: float
    bulk_quartic_coefficient: float
    elastic_l1_coefficient: float
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int
    pointwise_target: np.ndarray
    pointwise_prediction: np.ndarray
    pointwise_features: np.ndarray
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    independent_component_names: tuple[str, str] = ("Qxx", "Qxy")


@dataclass(frozen=True)
class BEQEquationLocalWeakFormFitResult2D:
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int
    local_target: np.ndarray
    local_prediction: np.ndarray
    local_features: np.ndarray
    patch_specs: tuple[LocalPatchSpec2D, ...]


@dataclass(frozen=True)
class BEPointwiseVelocityFitResult2D:
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int
    pointwise_target: np.ndarray
    pointwise_prediction: np.ndarray
    pointwise_features: np.ndarray


@dataclass(frozen=True)
class BELocalVelocityFitResult2D:
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int
    local_target: np.ndarray
    local_prediction: np.ndarray
    local_features: np.ndarray
    patch_specs: tuple[LocalPatchSpec2D, ...]


_Q_FEATURE_NAMES = (
    "material_q",
    "omega_q",
    "e_plain",
    "e_q",
    "q_colon_e_q",
    "flow_alignment",
    "bulk_linear",
    "bulk_quartic",
    "elastic_l1",
)

_Q_DEFAULT_FEATURE_NAMES = (
    "material_q",
    "omega_q",
    "e_plain",
    "q_colon_e_q",
    "bulk_linear",
    "bulk_quartic",
    "elastic_l1",
)


def _validate_2d_layout(
    velocity_field: np.ndarray,
    q_tensor_field: np.ndarray,
    spatial_axes: Sequence[int],
) -> tuple[int, int]:
    spatial_axes = tuple(int(axis) for axis in spatial_axes)
    if len(spatial_axes) != 2:
        raise ValueError("Pure 2D validation requires exactly two spatial axes.")
    if velocity_field.shape[-1] != 2:
        raise ValueError("Velocity must end with two components.")
    if q_tensor_field.shape[-2:] != (2, 2):
        raise ValueError("Q must end with shape (2, 2).")
    if velocity_field.shape[:-1] != q_tensor_field.shape[:-2]:
        raise ValueError("Velocity and Q must share the same time/space layout.")
    return spatial_axes


def _normalize_features(
    include_terms: Sequence[str] | None,
    valid_names: Sequence[str],
    default_names: Sequence[str],
) -> tuple[str, ...]:
    names = tuple(default_names if include_terms is None else include_terms)
    unknown = sorted(set(names) - set(valid_names))
    if unknown:
        raise ValueError(f"Unknown feature names {unknown}; valid names are {tuple(valid_names)}.")
    if len(names) != len(set(names)):
        raise ValueError("include_terms must not contain duplicate names.")
    if not names:
        raise ValueError("At least one feature must be included.")
    return names


def _time_staging(
    count: int,
    *,
    time_discretization: str,
    rhs_time_level: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    method = time_discretization.lower()
    rhs_time_level = rhs_time_level.lower()
    if rhs_time_level in {"n", "previous", "old"}:
        rhs_time_level = "previous"
    elif rhs_time_level in {"n+1", "current", "new"}:
        rhs_time_level = "current"
    else:
        raise ValueError("rhs_time_level must be previous/n or current/n+1.")

    if method in {"two_point", "forward", "adjacent"}:
        if count < 2:
            raise ValueError("At least two time slices are required.")
        target_indices = np.arange(count - 1, dtype=np.intp)
        rhs_indices = target_indices + (1 if rhs_time_level == "current" else 0)
        return target_indices, rhs_indices, "forward"
    if method in {"central", "centered"}:
        if count < 3:
            raise ValueError("Central time differences require at least three slices.")
        target_indices = np.arange(1, count - 1, dtype=np.intp)
        rhs_indices = target_indices + (1 if rhs_time_level == "current" else -1)
        return target_indices, rhs_indices, "central"
    raise ValueError("time_discretization must be two_point or central.")


def _spatial_slice(
    field: np.ndarray,
    *,
    spatial_axes: Sequence[int],
    sample_step: int | Sequence[int],
) -> np.ndarray:
    if isinstance(sample_step, (int, np.integer)):
        steps = (int(sample_step), int(sample_step))
    else:
        steps = tuple(int(value) for value in sample_step)
    if len(steps) != 2 or min(steps) < 1:
        raise ValueError("sample_step must be a positive int or length-2 sequence.")
    slices = [slice(None)] * field.ndim
    for axis, step in zip(spatial_axes, steps):
        slices[axis] = slice(None, None, step)
    return field[tuple(slices)]


def _q_features(
    velocity_rhs: np.ndarray,
    q_rhs: np.ndarray,
    *,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
    periodic: bool,
    spatial_derivative_method: str,
    include_terms: tuple[str, ...],
    flow_alignment_lambda: float,
) -> dict[str, np.ndarray]:
    non_advection_method = (
        "finite_difference"
        if spatial_derivative_method in {"upwind", "upwind3", "third_order_upwind"}
        else spatial_derivative_method
    )
    features: dict[str, np.ndarray] = {}
    if "material_q" in include_terms:
        features["material_q"] = q_material_derivative(
            velocity_rhs,
            q_rhs,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
    interaction_names = {
        "omega_q",
        "e_plain",
        "e_q",
        "q_colon_e_q",
        "flow_alignment",
    }
    if interaction_names.intersection(include_terms):
        e_tensor, omega_tensor = velocity_shear_and_rotation_tensors(
            velocity_rhs,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=non_advection_method,
        )
        if "omega_q" in include_terms:
            features["omega_q"] = q_omega_interaction(q_rhs, omega_tensor)
        if "e_plain" in include_terms:
            features["e_plain"] = q_e_plain_interaction(e_tensor)
        if "e_q" in include_terms:
            features["e_q"] = q_e_interaction(q_rhs, e_tensor)
        if "q_colon_e_q" in include_terms:
            features["q_colon_e_q"] = q_colon_e_q_interaction(q_rhs, e_tensor)
        if "flow_alignment" in include_terms:
            features["flow_alignment"] = q_flow_alignment(
                q_rhs,
                e_tensor,
                omega_tensor,
                lambda_flow_alignment=flow_alignment_lambda,
            )
    if "bulk_linear" in include_terms:
        features["bulk_linear"] = bulk_linear_term(q_rhs)
    if "bulk_quartic" in include_terms:
        features["bulk_quartic"] = bulk_quartic_term(q_rhs)
    if "elastic_l1" in include_terms:
        features["elastic_l1"] = elastic_molecular_field_l1(
            q_rhs,
            l1=1.0,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=non_advection_method,
        )
    return features


def fit_be_q_equation_pointwise(
    velocity_field: np.ndarray,
    q_tensor_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    time_discretization: str = "two_point",
    rhs_time_level: str = "current",
    include_terms: Sequence[str] | None = None,
    flow_alignment_lambda: float = 1.0,
    sample_step: int | Sequence[int] = 1,
    threshold: float = 1e-10,
    optimizer_alpha: float = 0.0,
    normalize_columns: bool = False,
) -> BEQEquationPointwiseFitResult2D:
    spatial_axes = _validate_2d_layout(velocity_field, q_tensor_field, spatial_axes)
    feature_names = _normalize_features(
        include_terms, _Q_FEATURE_NAMES, _Q_DEFAULT_FEATURE_NAMES
    )
    target_indices, rhs_indices, derivative_method = _time_staging(
        q_tensor_field.shape[time_axis],
        time_discretization=time_discretization,
        rhs_time_level=rhs_time_level,
    )
    dt_q = q_time_derivative(
        q_tensor_field,
        dt=dt,
        time_axis=time_axis,
        time_derivative_method=derivative_method,
    )
    target = np.take(dt_q, target_indices, axis=time_axis)
    velocity_rhs = np.take(velocity_field, rhs_indices, axis=time_axis)
    q_rhs = np.take(q_tensor_field, rhs_indices, axis=time_axis)
    feature_fields = _q_features(
        velocity_rhs,
        q_rhs,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
        include_terms=feature_names,
        flow_alignment_lambda=flow_alignment_lambda,
    )

    target = extract_independent_q_components(
        _spatial_slice(target, spatial_axes=spatial_axes, sample_step=sample_step)
    )
    regression_features = tuple(
        extract_independent_q_components(
            _spatial_slice(
                feature_fields[name], spatial_axes=spatial_axes, sample_step=sample_step
            )
        )
        for name in feature_names
    )
    coefficients, prediction, r2, rmse, relative_residual = fit_linear_term_model(
        target,
        regression_features,
        threshold=threshold,
        alpha=optimizer_alpha,
        normalize_columns=normalize_columns,
    )
    coefficient_by_name = dict(zip(feature_names, coefficients))
    return BEQEquationPointwiseFitResult2D(
        material_q_coefficient=float(coefficient_by_name.get("material_q", 0.0)),
        omega_q_coefficient=float(coefficient_by_name.get("omega_q", 0.0)),
        e_plain_coefficient=float(coefficient_by_name.get("e_plain", 0.0)),
        e_q_coefficient=float(coefficient_by_name.get("e_q", 0.0)),
        q_colon_e_q_coefficient=float(coefficient_by_name.get("q_colon_e_q", 0.0)),
        flow_alignment_coefficient=float(coefficient_by_name.get("flow_alignment", 0.0)),
        bulk_linear_coefficient=float(coefficient_by_name.get("bulk_linear", 0.0)),
        bulk_quartic_coefficient=float(coefficient_by_name.get("bulk_quartic", 0.0)),
        elastic_l1_coefficient=float(coefficient_by_name.get("elastic_l1", 0.0)),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(target.size),
        pointwise_target=target,
        pointwise_prediction=prediction.reshape(target.shape),
        pointwise_features=np.stack(regression_features, axis=-1),
        coefficients=coefficients,
        feature_names=feature_names,
    )


def generate_local_patch_specs(
    shape: Sequence[int],
    *,
    half_widths: tuple[int, int] = (3, 3),
    strides: tuple[int, int] = (4, 4),
) -> tuple[LocalPatchSpec2D, ...]:
    nx, ny = (int(value) for value in shape)
    hx, hy = half_widths
    sx, sy = strides
    if min(hx, hy, sx, sy) < 1:
        raise ValueError("Patch half-widths and strides must be positive.")
    patches = []
    for cx in range(hx, nx - hx, sx):
        for cy in range(hy, ny - hy, sy):
            patches.append(LocalPatchSpec2D((cx, cy), (hx, hy)))
    if not patches:
        raise ValueError("No local patches fit inside the selected 2D domain.")
    return tuple(patches)


def _patch_slices(patch: LocalPatchSpec2D, spatial_axes: Sequence[int], ndim: int) -> tuple[slice, ...]:
    slices = [slice(None)] * ndim
    for center, half_width, axis in zip(
        patch.center_indices, patch.half_widths, spatial_axes
    ):
        slices[axis] = slice(center - half_width, center + half_width + 1)
    return tuple(slices)


def _scalar_bump(patch: LocalPatchSpec2D) -> np.ndarray:
    hx, hy = patch.half_widths
    x = np.arange(-hx, hx + 1, dtype=float) / hx
    y = np.arange(-hy, hy + 1, dtype=float) / hy
    bx = (1.0 - x**2) ** 2
    by = (1.0 - y**2) ** 2
    return bx[:, None] * by[None, :]


def _weighted_spatial_integral(
    field: np.ndarray,
    weight: np.ndarray,
    *,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
) -> np.ndarray:
    reshape = [1] * field.ndim
    reshape[spatial_axes[0]] = weight.shape[0]
    reshape[spatial_axes[1]] = weight.shape[1]
    weighted = field * weight.reshape(reshape)
    return np.sum(weighted, axis=tuple(spatial_axes)) * float(np.prod(spacings))


def fit_be_q_equation_local_weak_form(
    velocity_field: np.ndarray,
    q_tensor_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    patch_half_widths: tuple[int, int] = (3, 3),
    patch_strides: tuple[int, int] = (4, 4),
    patch_specs: Sequence[LocalPatchSpec2D] | None = None,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    time_discretization: str = "two_point",
    rhs_time_level: str = "current",
    include_terms: Sequence[str] | None = None,
    flow_alignment_lambda: float = 1.0,
    threshold: float = 1e-10,
    optimizer_alpha: float = 0.0,
    normalize_columns: bool = False,
) -> BEQEquationLocalWeakFormFitResult2D:
    spatial_axes = _validate_2d_layout(velocity_field, q_tensor_field, spatial_axes)
    feature_names = _normalize_features(
        include_terms, _Q_FEATURE_NAMES, _Q_DEFAULT_FEATURE_NAMES
    )
    target_indices, rhs_indices, derivative_method = _time_staging(
        q_tensor_field.shape[time_axis],
        time_discretization=time_discretization,
        rhs_time_level=rhs_time_level,
    )
    target = np.take(
        q_time_derivative(
            q_tensor_field,
            dt=dt,
            time_axis=time_axis,
            time_derivative_method=derivative_method,
        ),
        target_indices,
        axis=time_axis,
    )
    velocity_rhs = np.take(velocity_field, rhs_indices, axis=time_axis)
    q_rhs = np.take(q_tensor_field, rhs_indices, axis=time_axis)
    feature_fields = _q_features(
        velocity_rhs,
        q_rhs,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
        include_terms=feature_names,
        flow_alignment_lambda=flow_alignment_lambda,
    )
    spatial_shape = tuple(q_tensor_field.shape[axis] for axis in spatial_axes)
    if patch_specs is None:
        patch_specs = generate_local_patch_specs(
            spatial_shape,
            half_widths=patch_half_widths,
            strides=patch_strides,
        )
    else:
        patch_specs = tuple(patch_specs)

    local_target = []
    local_features = {name: [] for name in feature_names}
    for patch in patch_specs:
        weight = _scalar_bump(patch)
        target_patch = target[_patch_slices(patch, spatial_axes, target.ndim)]
        target_integral = _weighted_spatial_integral(
            target_patch,
            weight,
            spacings=spacings,
            spatial_axes=spatial_axes,
        )
        local_target.append(extract_independent_q_components(target_integral))
        for name in feature_names:
            field = feature_fields[name]
            field_patch = field[_patch_slices(patch, spatial_axes, field.ndim)]
            integral = _weighted_spatial_integral(
                field_patch,
                weight,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
            local_features[name].append(extract_independent_q_components(integral))

    target_array = np.stack(local_target, axis=1)
    feature_arrays = tuple(np.stack(local_features[name], axis=1) for name in feature_names)
    coefficients, prediction, r2, rmse, relative_residual = fit_linear_term_model(
        target_array,
        feature_arrays,
        threshold=threshold,
        alpha=optimizer_alpha,
        normalize_columns=normalize_columns,
    )
    return BEQEquationLocalWeakFormFitResult2D(
        coefficients=coefficients,
        feature_names=feature_names,
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(target_array.size),
        local_target=target_array,
        local_prediction=prediction.reshape(target_array.shape),
        local_features=np.stack(feature_arrays, axis=-1),
        patch_specs=tuple(patch_specs),
    )


def _velocity_feature_fields(
    velocity_rhs: np.ndarray,
    q_rhs: np.ndarray,
    *,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
    periodic: bool,
    spatial_derivative_method: str,
    include_terms: tuple[str, ...],
    pressure_rhs: np.ndarray | None,
    density_rhs: np.ndarray | float,
    passive_parameters: tuple[float, float, float, float] | None,
) -> dict[str, np.ndarray]:
    non_advection_method = (
        "finite_difference"
        if spatial_derivative_method in {"upwind", "upwind3", "third_order_upwind"}
        else spatial_derivative_method
    )
    features: dict[str, np.ndarray] = {}
    if "linear_velocity" in include_terms:
        features["linear_velocity"] = velocity_rhs
    if "quadratic_velocity" in include_terms:
        features["quadratic_velocity"] = aligned_quadratic_velocity(velocity_rhs)
    if "material" in include_terms:
        features["material"] = velocity_material_derivative(
            velocity_rhs,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
    if "laplacian" in include_terms:
        features["laplacian"] = velocity_laplacian(
            velocity_rhs,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=non_advection_method,
        )
    if "q_divergence" in include_terms:
        features["q_divergence"] = active_q_divergence_force(
            q_rhs,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=non_advection_method,
        )
    if "pressure_gradient_over_density" in include_terms:
        if pressure_rhs is None:
            raise ValueError("pressure_field is required for pressure-gradient fitting.")
        pressure_grad = np.stack(
            gradient(
                pressure_rhs,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                method=non_advection_method,
            ),
            axis=-1,
        )
        density = np.asarray(density_rhs, dtype=float)
        features["pressure_gradient_over_density"] = pressure_grad / density[..., None]
    if "passive_backflow" in include_terms:
        if passive_parameters is None:
            raise ValueError(
                "passive_parameters=(a2, a4, kappa, flow_alignment_xi) is required."
            )
        a2, a4, kappa, flow_alignment_xi = passive_parameters
        features["passive_backflow"] = passive_backflow_force(
            q_rhs,
            a2=a2,
            a4=a4,
            kappa=kappa,
            flow_alignment_xi=flow_alignment_xi,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=non_advection_method,
        )
    return features


def fit_be_equation_velocity_pointwise(
    velocity_field: np.ndarray,
    q_tensor_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    pressure_field: np.ndarray | None = None,
    density_field: np.ndarray | float = 1.0,
    passive_parameters: tuple[float, float, float, float] | None = None,
    include_terms: Sequence[str] | None = None,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    time_discretization: str = "two_point",
    rhs_time_level: str = "current",
    sample_step: int | Sequence[int] = 1,
    threshold: float = 1e-10,
    optimizer_alpha: float = 0.0,
    normalize_columns: bool = False,
) -> BEPointwiseVelocityFitResult2D:
    spatial_axes = _validate_2d_layout(velocity_field, q_tensor_field, spatial_axes)
    valid_names = (
        "linear_velocity",
        "quadratic_velocity",
        "material",
        "laplacian",
        "pressure_gradient_over_density",
        "q_divergence",
        "passive_backflow",
    )
    default_names = ["material", "laplacian", "q_divergence"]
    if pressure_field is not None:
        default_names.append("pressure_gradient_over_density")
    if passive_parameters is not None:
        default_names.append("passive_backflow")
    feature_names = _normalize_features(include_terms, valid_names, default_names)
    target_indices, rhs_indices, derivative_method = _time_staging(
        velocity_field.shape[time_axis],
        time_discretization=time_discretization,
        rhs_time_level=rhs_time_level,
    )
    target = np.take(
        velocity_time_derivative_2d(
            velocity_field,
            dt=dt,
            time_axis=time_axis,
            time_derivative_method=derivative_method,
        ),
        target_indices,
        axis=time_axis,
    )
    velocity_rhs = np.take(velocity_field, rhs_indices, axis=time_axis)
    q_rhs = np.take(q_tensor_field, rhs_indices, axis=time_axis)
    pressure_rhs = (
        None if pressure_field is None else np.take(pressure_field, rhs_indices, axis=time_axis)
    )
    if np.ndim(density_field) == 0:
        density_rhs: np.ndarray | float = float(density_field)
    else:
        density_rhs = np.take(np.asarray(density_field), rhs_indices, axis=time_axis)
    features = _velocity_feature_fields(
        velocity_rhs,
        q_rhs,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
        include_terms=feature_names,
        pressure_rhs=pressure_rhs,
        density_rhs=density_rhs,
        passive_parameters=passive_parameters,
    )
    target = _spatial_slice(target, spatial_axes=spatial_axes, sample_step=sample_step)
    regression_features = tuple(
        _spatial_slice(features[name], spatial_axes=spatial_axes, sample_step=sample_step)
        for name in feature_names
    )
    coefficients, prediction, r2, rmse, relative_residual = fit_linear_term_model(
        target,
        regression_features,
        threshold=threshold,
        alpha=optimizer_alpha,
        normalize_columns=normalize_columns,
    )
    return BEPointwiseVelocityFitResult2D(
        coefficients=coefficients,
        feature_names=feature_names,
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(target.size),
        pointwise_target=target,
        pointwise_prediction=prediction.reshape(target.shape),
        pointwise_features=np.stack(regression_features, axis=-1),
    )


def _divergence_free_patch_test_field(
    patch: LocalPatchSpec2D,
    spacings: Sequence[float],
) -> np.ndarray:
    hx, hy = patch.half_widths
    dx, dy = (float(value) for value in spacings)
    x = np.arange(-hx, hx + 1, dtype=float) / hx
    y = np.arange(-hy, hy + 1, dtype=float) / hy
    bx = (1.0 - x**2) ** 2
    by = (1.0 - y**2) ** 2
    dbx_dx = -4.0 * x * (1.0 - x**2) / (hx * dx)
    dby_dy = -4.0 * y * (1.0 - y**2) / (hy * dy)
    phi_x = bx[:, None] * dby_dy[None, :]
    phi_y = -dbx_dx[:, None] * by[None, :]
    return np.stack((phi_x, phi_y), axis=-1)


def _vector_patch_inner_product(
    vector_field: np.ndarray,
    test_field: np.ndarray,
    *,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
) -> np.ndarray:
    reshape = [1] * vector_field.ndim
    reshape[spatial_axes[0]] = test_field.shape[0]
    reshape[spatial_axes[1]] = test_field.shape[1]
    reshape[-1] = 2
    pointwise = np.sum(vector_field * test_field.reshape(reshape), axis=-1)
    return np.sum(pointwise, axis=tuple(spatial_axes)) * float(np.prod(spacings))


def fit_be_equation_velocity_local_weak_form(
    velocity_field: np.ndarray,
    q_tensor_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    passive_parameters: tuple[float, float, float, float] | None = None,
    include_terms: Sequence[str] | None = None,
    patch_half_widths: tuple[int, int] = (3, 3),
    patch_strides: tuple[int, int] = (4, 4),
    patch_specs: Sequence[LocalPatchSpec2D] | None = None,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    time_discretization: str = "two_point",
    rhs_time_level: str = "current",
    threshold: float = 1e-10,
    optimizer_alpha: float = 0.0,
    normalize_columns: bool = False,
) -> BELocalVelocityFitResult2D:
    """Fit the 2D velocity equation with compact divergence-free test fields.

    Pressure is omitted because each local test field is generated as the curl
    of a scalar stream function and is therefore divergence free.
    """
    spatial_axes = _validate_2d_layout(velocity_field, q_tensor_field, spatial_axes)
    valid_names = (
        "linear_velocity",
        "quadratic_velocity",
        "material",
        "laplacian",
        "q_divergence",
        "passive_backflow",
    )
    default_names = ["material", "laplacian", "q_divergence"]
    if passive_parameters is not None:
        default_names.append("passive_backflow")
    feature_names = _normalize_features(include_terms, valid_names, default_names)
    target_indices, rhs_indices, derivative_method = _time_staging(
        velocity_field.shape[time_axis],
        time_discretization=time_discretization,
        rhs_time_level=rhs_time_level,
    )
    target = np.take(
        velocity_time_derivative_2d(
            velocity_field,
            dt=dt,
            time_axis=time_axis,
            time_derivative_method=derivative_method,
        ),
        target_indices,
        axis=time_axis,
    )
    velocity_rhs = np.take(velocity_field, rhs_indices, axis=time_axis)
    q_rhs = np.take(q_tensor_field, rhs_indices, axis=time_axis)
    features = _velocity_feature_fields(
        velocity_rhs,
        q_rhs,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
        include_terms=feature_names,
        pressure_rhs=None,
        density_rhs=1.0,
        passive_parameters=passive_parameters,
    )
    spatial_shape = tuple(q_tensor_field.shape[axis] for axis in spatial_axes)
    if patch_specs is None:
        patch_specs = generate_local_patch_specs(
            spatial_shape,
            half_widths=patch_half_widths,
            strides=patch_strides,
        )
    else:
        patch_specs = tuple(patch_specs)

    local_target = []
    local_features = {name: [] for name in feature_names}
    for patch in patch_specs:
        test_field = _divergence_free_patch_test_field(patch, spacings)
        target_patch = target[_patch_slices(patch, spatial_axes, target.ndim)]
        local_target.append(
            _vector_patch_inner_product(
                target_patch,
                test_field,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
        )
        for name in feature_names:
            field = features[name]
            field_patch = field[_patch_slices(patch, spatial_axes, field.ndim)]
            local_features[name].append(
                _vector_patch_inner_product(
                    field_patch,
                    test_field,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
    target_array = np.stack(local_target, axis=1)
    feature_arrays = tuple(np.stack(local_features[name], axis=1) for name in feature_names)
    coefficients, prediction, r2, rmse, relative_residual = fit_linear_term_model(
        target_array,
        feature_arrays,
        threshold=threshold,
        alpha=optimizer_alpha,
        normalize_columns=normalize_columns,
    )
    return BELocalVelocityFitResult2D(
        coefficients=coefficients,
        feature_names=feature_names,
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(target_array.size),
        local_target=target_array,
        local_prediction=prediction.reshape(target_array.shape),
        local_features=np.stack(feature_arrays, axis=-1),
        patch_specs=tuple(patch_specs),
    )


__all__ = [
    "BELocalVelocityFitResult2D",
    "BEPointwiseVelocityFitResult2D",
    "BEQEquationLocalWeakFormFitResult2D",
    "BEQEquationPointwiseFitResult2D",
    "LocalPatchSpec2D",
    "fit_be_equation_velocity_local_weak_form",
    "fit_be_equation_velocity_pointwise",
    "fit_be_q_equation_local_weak_form",
    "fit_be_q_equation_pointwise",
    "generate_local_patch_specs",
]
