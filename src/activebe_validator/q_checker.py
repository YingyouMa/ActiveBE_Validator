"""Local weak-form coefficient checks for Q-tensor dynamics."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._regression import fit_linear_term_model as _fit_linear_term_model
from .ns_checker import (
    LocalPatchSpec,
    _build_local_bump,
    _crop_patch_with_halo,
    _validate_local_test_field_mode,
    generate_local_patch_specs,
    integrate_over_space,
)
from .q_terms import (
    bulk_cubic_term,
    bulk_linear_term,
    bulk_quadratic_term,
    elastic_molecular_field_l1,
    q_colon_e_q_interaction,
    q_e_interaction,
    q_e_plain_interaction,
    q_flow_alignment,
    q_material_derivative,
    q_omega_interaction,
    q_time_derivative,
)
from .velocity_terms import velocity_shear_and_rotation_tensors


@dataclass(frozen=True)
class BEQEquationLocalWeakFormFitResult:
    """Summary of a local weak-form fit for a simplified Q equation."""

    dt_q_coefficient: float
    material_q_coefficient: float
    omega_q_coefficient: float
    e_plain_coefficient: float
    e_q_coefficient: float
    q_colon_e_q_coefficient: float
    flow_alignment_coefficient: float
    bulk_linear_coefficient: float
    bulk_quadratic_coefficient: float
    bulk_cubic_coefficient: float
    elastic_l1_coefficient: float
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int
    local_target: np.ndarray
    local_prediction: np.ndarray
    local_features: np.ndarray
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    patch_specs: tuple[LocalPatchSpec, ...]
    target_name: str


@dataclass(frozen=True)
class BEQEquationPointwiseFitResult:
    """Summary of a pointwise fit for a simplified Q equation."""

    dt_q_coefficient: float
    material_q_coefficient: float
    omega_q_coefficient: float
    e_plain_coefficient: float
    e_q_coefficient: float
    q_colon_e_q_coefficient: float
    flow_alignment_coefficient: float
    bulk_linear_coefficient: float
    bulk_quadratic_coefficient: float
    bulk_cubic_coefficient: float
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
    independent_component_names: tuple[str, ...]
    sample_slices: tuple[object, object, object] | None = None


_Q_EQUATION_FEATURE_NAMES = (
    "material_q",
    "omega_q",
    "e_plain",
    "e_q",
    "q_colon_e_q",
    "flow_alignment",
    "bulk_linear",
    "bulk_quadratic",
    "bulk_cubic",
    "elastic_l1",
)
_Q_EQUATION_DEFAULT_FEATURE_NAMES = (
    "material_q",
    "omega_q",
    "e_q",
    "bulk_linear",
    "bulk_quadratic",
    "bulk_cubic",
    "elastic_l1",
)


def _normalize_q_advection_derivative_method(method: str) -> str:
    aliases = {
        "centered": "finite_difference",
        "central": "finite_difference",
        "finite_difference": "finite_difference",
        "spectral": "spectral",
        "upwind": "upwind3",
        "upwind3": "upwind3",
        "third_order_upwind": "upwind3",
    }
    normalized = aliases.get(method.lower(), method.lower())
    if normalized not in {"finite_difference", "spectral", "upwind3"}:
        raise ValueError(
            "spatial_derivative_method for Q fits must be 'finite_difference', "
            f"'spectral', or 'upwind3', got {method!r}."
        )
    return normalized


def _support_derivative_method_for_non_advection(method: str) -> str:
    """Return the derivative method used by non-advective Q terms."""
    if method == "upwind3":
        return "finite_difference"
    return method


def _validate_time_discretization(time_discretization: str) -> str:
    aliases = {
        "adjacent": "two_point",
        "forward": "two_point",
        "euler": "two_point",
        "two_point": "two_point",
        "central": "central",
        "centered": "central",
    }
    normalized = aliases.get(time_discretization.lower(), time_discretization.lower())
    if normalized not in {"two_point", "central"}:
        raise ValueError(
            "time_discretization must be either 'two_point' or 'central', "
            f"got {time_discretization!r}."
        )
    return normalized


def _validate_rhs_time_level(name: str, value: str) -> str:
    aliases = {
        "n": "previous",
        "old": "previous",
        "prev": "previous",
        "previous": "previous",
        "explicit": "previous",
        "n+1": "current",
        "new": "current",
        "current": "current",
        "implicit": "current",
    }
    normalized = aliases.get(value.lower(), value.lower())
    if normalized not in {"previous", "current"}:
        raise ValueError(
            f"{name} must be either 'previous' or 'current', got {value!r}."
        )
    return normalized


def _resolve_velocity_q_time_levels(
    *,
    v_time_level: str | None,
    q_time_level: str,
) -> tuple[str, str]:
    """Resolve explicit separate v/Q time levels."""
    resolved_v_time_level = "current" if v_time_level is None else v_time_level
    return resolved_v_time_level, q_time_level


def _time_indices_for_q_regression(
    count: int,
    *,
    use_interior_time_only: bool,
    time_discretization: str,
    v_time_level: str,
    q_time_level: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return target, velocity-feature, and Q-feature time indices for Q fits."""
    time_discretization = _validate_time_discretization(time_discretization)
    if count < 2:
        raise ValueError("Q time-derivative fits require at least two time slices.")

    if time_discretization == "central" and count < 3:
        raise ValueError("Central time differences require at least three time slices.")

    if time_discretization == "two_point":
        target_indices = np.arange(0, count - 1, dtype=np.intp)
        if use_interior_time_only and target_indices.size > 1:
            target_indices = target_indices[1:]
        velocity_indices = target_indices
        if v_time_level == "current":
            velocity_indices = target_indices + 1
        q_indices = target_indices
        if q_time_level == "current":
            q_indices = target_indices + 1
        return target_indices, velocity_indices, q_indices, "forward"

    target_indices = np.arange(1, count - 1, dtype=np.intp)
    if use_interior_time_only:
        target_indices = target_indices.copy()
    if target_indices.size == 0:
        raise ValueError(
            "The selected Q time-staging settings have no usable time samples."
        )

    velocity_indices = target_indices - 1
    if v_time_level == "current":
        velocity_indices = target_indices + 1

    q_indices = target_indices - 1
    if q_time_level == "current":
        q_indices = target_indices + 1

    return target_indices, velocity_indices, q_indices, "central"


def _validate_included_feature_names(
    include_terms: Sequence[str] | None,
    valid_feature_names: tuple[str, ...],
    default_feature_names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if include_terms is None:
        if default_feature_names is not None:
            return default_feature_names
        return valid_feature_names

    included_feature_names = tuple(include_terms)
    if not included_feature_names:
        raise ValueError("include_terms must contain at least one term name.")

    unknown_terms = sorted(set(included_feature_names) - set(valid_feature_names))
    if unknown_terms:
        valid_terms = ", ".join(valid_feature_names)
        raise ValueError(
            f"Unknown include_terms entries {unknown_terms}; valid terms are: {valid_terms}."
        )

    if len(set(included_feature_names)) != len(included_feature_names):
        raise ValueError("include_terms must not contain duplicate term names.")

    return included_feature_names


def _normalize_pointwise_sample_slices(
    sample_slices: Sequence[object] | None,
    sample_step: int | Sequence[int] | None,
    spatial_shape: tuple[int, int, int],
) -> tuple[object, object, object] | None:
    if sample_slices is not None and sample_step is not None:
        raise ValueError("sample_slices and sample_step cannot both be provided.")

    if sample_slices is None and sample_step is None:
        return None

    if sample_step is not None:
        if isinstance(sample_step, (int, np.integer)):
            step = int(sample_step)
            sample_steps = (step, step, step)
        else:
            sample_steps = tuple(int(step) for step in sample_step)
            if len(sample_steps) != 3:
                raise ValueError("sample_step must be an int or a length-3 sequence.")
        if any(step <= 0 for step in sample_steps):
            raise ValueError("sample_step entries must be positive integers.")
        return tuple(slice(None, None, step) for step in sample_steps)

    raw_slices = tuple(sample_slices)
    if len(raw_slices) != 3:
        raise ValueError(
            "sample_slices must contain exactly one selector per spatial axis."
        )

    normalized_slices: list[object] = []
    for selector, axis_length in zip(raw_slices, spatial_shape):
        if isinstance(selector, slice):
            normalized_slices.append(selector)
            continue
        selector_array = np.atleast_1d(np.asarray(selector))
        if not np.issubdtype(selector_array.dtype, np.integer):
            raise TypeError(
                "sample_slices entries must be slices or integer index arrays."
            )
        if selector_array.size and (
            np.any(selector_array >= axis_length)
            or np.any(selector_array < -axis_length)
        ):
            raise IndexError(
                "sample_slices contains an index outside the spatial domain."
            )
        normalized_slices.append(selector_array)

    return tuple(normalized_slices)


def _apply_pointwise_sample_slices(
    field: np.ndarray,
    sample_slices: tuple[object, object, object] | None,
    spatial_axes: tuple[int, int, int],
) -> np.ndarray:
    if sample_slices is None:
        return field

    sampled_field = field
    for axis, selector in zip(spatial_axes, sample_slices):
        normalized_axis = axis % field.ndim
        if isinstance(selector, slice):
            indexer: list[object] = [slice(None)] * sampled_field.ndim
            indexer[normalized_axis] = selector
            sampled_field = sampled_field[tuple(indexer)]
        else:
            sampled_field = np.take(sampled_field, selector, axis=normalized_axis)
    return sampled_field


def _pointwise_sample_indices(
    sample_slices: tuple[object, object, object],
    spatial_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices: list[np.ndarray] = []
    for selector, axis_length in zip(sample_slices, spatial_shape):
        if isinstance(selector, slice):
            axis_indices = np.arange(axis_length, dtype=np.intp)[selector]
        else:
            axis_indices = np.asarray(selector, dtype=np.intp)
            axis_indices = np.where(
                axis_indices < 0, axis_indices + axis_length, axis_indices
            )
        if axis_indices.size == 0:
            raise ValueError(
                "Pointwise sampling selected no grid points on a spatial axis."
            )
        indices.append(axis_indices)
    return tuple(indices)


def _take_spatial_points(
    field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    spatial_axes: tuple[int, int, int],
) -> np.ndarray:
    sampled_field = field
    for axis, indices in zip(spatial_axes, sample_indices):
        sampled_field = np.take(sampled_field, indices, axis=axis % sampled_field.ndim)
    return sampled_field


def _neighbor_indices(
    indices: np.ndarray,
    axis_length: int,
    offset: int,
    periodic: bool,
) -> np.ndarray:
    if periodic:
        return (indices + offset) % axis_length
    return np.clip(indices + offset, 0, axis_length - 1)


def _replace_axis_values(
    target: np.ndarray,
    values: np.ndarray,
    axis: int,
    mask: np.ndarray,
) -> None:
    if not np.any(mask):
        return
    indexer: list[object] = [slice(None)] * target.ndim
    indexer[axis % target.ndim] = mask
    target[tuple(indexer)] = values[tuple(indexer)]


def _broadcast_axis_mask(
    axis_mask: np.ndarray,
    target_ndim: int,
    axis: int,
) -> np.ndarray:
    reshaped = [1] * target_ndim
    reshaped[axis % target_ndim] = axis_mask.size
    return np.asarray(axis_mask, dtype=bool).reshape(reshaped)


def _sampled_first_derivative(
    field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    spatial_axes: tuple[int, int, int],
    derivative_axis_index: int,
    spacing: float,
    periodic: bool,
) -> np.ndarray:
    axis_length = field.shape[spatial_axes[derivative_axis_index]]
    center_indices = sample_indices[derivative_axis_index]
    left_indices = _neighbor_indices(center_indices, axis_length, -1, periodic)
    right_indices = _neighbor_indices(center_indices, axis_length, 1, periodic)

    left_sample = list(sample_indices)
    right_sample = list(sample_indices)
    left_sample[derivative_axis_index] = left_indices
    right_sample[derivative_axis_index] = right_indices

    left = _take_spatial_points(field, tuple(left_sample), spatial_axes)
    right = _take_spatial_points(field, tuple(right_sample), spatial_axes)
    derivative = (right - left) / (2.0 * spacing)

    if not periodic:
        center = _take_spatial_points(field, sample_indices, spatial_axes)
        start_mask = center_indices == 0
        end_mask = center_indices == axis_length - 1
        start_derivative = (right - center) / spacing
        end_derivative = (center - left) / spacing
        output_axis = spatial_axes[derivative_axis_index] % derivative.ndim
        _replace_axis_values(derivative, start_derivative, output_axis, start_mask)
        _replace_axis_values(derivative, end_derivative, output_axis, end_mask)

    return derivative


def _sampled_second_derivative(
    field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    spatial_axes: tuple[int, int, int],
    derivative_axis_index: int,
    spacing: float,
    periodic: bool,
) -> np.ndarray:
    axis_length = field.shape[spatial_axes[derivative_axis_index]]
    center_indices = sample_indices[derivative_axis_index]
    center = _take_spatial_points(field, sample_indices, spatial_axes)

    left_indices = _neighbor_indices(center_indices, axis_length, -1, periodic)
    right_indices = _neighbor_indices(center_indices, axis_length, 1, periodic)
    left_sample = list(sample_indices)
    right_sample = list(sample_indices)
    left_sample[derivative_axis_index] = left_indices
    right_sample[derivative_axis_index] = right_indices
    left = _take_spatial_points(field, tuple(left_sample), spatial_axes)
    right = _take_spatial_points(field, tuple(right_sample), spatial_axes)
    derivative = (right - 2.0 * center + left) / (spacing**2)

    if not periodic:
        if axis_length < 3:
            raise ValueError(
                "Non-periodic second derivatives require at least 3 grid points."
            )
        output_axis = spatial_axes[derivative_axis_index] % derivative.ndim
        start_mask = center_indices == 0
        end_mask = center_indices == axis_length - 1

        next1_sample = list(sample_indices)
        next2_sample = list(sample_indices)
        next1_sample[derivative_axis_index] = np.clip(
            center_indices + 1, 0, axis_length - 1
        )
        next2_sample[derivative_axis_index] = np.clip(
            center_indices + 2, 0, axis_length - 1
        )
        next1 = _take_spatial_points(field, tuple(next1_sample), spatial_axes)
        next2 = _take_spatial_points(field, tuple(next2_sample), spatial_axes)
        start_derivative = (next2 - 2.0 * next1 + center) / (spacing**2)

        prev1_sample = list(sample_indices)
        prev2_sample = list(sample_indices)
        prev1_sample[derivative_axis_index] = np.clip(
            center_indices - 1, 0, axis_length - 1
        )
        prev2_sample[derivative_axis_index] = np.clip(
            center_indices - 2, 0, axis_length - 1
        )
        prev1 = _take_spatial_points(field, tuple(prev1_sample), spatial_axes)
        prev2 = _take_spatial_points(field, tuple(prev2_sample), spatial_axes)
        end_derivative = (center - 2.0 * prev1 + prev2) / (spacing**2)

        _replace_axis_values(derivative, start_derivative, output_axis, start_mask)
        _replace_axis_values(derivative, end_derivative, output_axis, end_mask)

    return derivative


def _sampled_tensor_laplacian(
    tensor_field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int],
    periodic: bool,
) -> np.ndarray:
    result = np.zeros_like(
        _take_spatial_points(tensor_field, sample_indices, spatial_axes)
    )
    for derivative_axis_index, spacing in enumerate(spacings):
        result += _sampled_second_derivative(
            tensor_field,
            sample_indices,
            spatial_axes,
            derivative_axis_index,
            spacing,
            periodic,
        )
    return result


def _sampled_q_material_derivative(
    velocity_field: np.ndarray,
    q_tensor_field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int],
    periodic: bool,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    spatial_derivative_method = _normalize_q_advection_derivative_method(
        spatial_derivative_method
    )
    velocity_center = _take_spatial_points(velocity_field, sample_indices, spatial_axes)
    result = np.zeros_like(
        _take_spatial_points(q_tensor_field, sample_indices, spatial_axes)
    )
    for component_index, spacing in enumerate(spacings):
        velocity_component = velocity_center[..., component_index]
        if spatial_derivative_method == "upwind3":
            q_partial = _sampled_first_derivative(
                q_tensor_field,
                sample_indices,
                spatial_axes,
                component_index,
                spacing,
                periodic,
            )
            axis_length = q_tensor_field.shape[spatial_axes[component_index]]
            center_indices = sample_indices[component_index]
            center = _take_spatial_points(q_tensor_field, sample_indices, spatial_axes)

            prev1_sample = list(sample_indices)
            prev1_sample[component_index] = _neighbor_indices(
                center_indices, axis_length, -1, periodic
            )
            prev2_sample = list(sample_indices)
            prev2_sample[component_index] = _neighbor_indices(
                center_indices, axis_length, -2, periodic
            )
            next1_sample = list(sample_indices)
            next1_sample[component_index] = _neighbor_indices(
                center_indices, axis_length, 1, periodic
            )
            next2_sample = list(sample_indices)
            next2_sample[component_index] = _neighbor_indices(
                center_indices, axis_length, 2, periodic
            )

            prev1 = _take_spatial_points(q_tensor_field, tuple(prev1_sample), spatial_axes)
            prev2 = _take_spatial_points(q_tensor_field, tuple(prev2_sample), spatial_axes)
            next1 = _take_spatial_points(q_tensor_field, tuple(next1_sample), spatial_axes)
            next2 = _take_spatial_points(q_tensor_field, tuple(next2_sample), spatial_axes)

            positive_derivative = (
                2.0 * next1 + 3.0 * center - 6.0 * prev1 + prev2
            ) / (6.0 * spacing)
            negative_derivative = (
                -next2 + 6.0 * next1 - 3.0 * center - 2.0 * prev1
            ) / (6.0 * spacing)

            positive_mask = velocity_component >= 0.0
            negative_mask = velocity_component < 0.0
            if not periodic:
                output_axis = spatial_axes[component_index] % positive_mask.ndim
                positive_interior = _broadcast_axis_mask(
                    (center_indices >= 2) & (center_indices <= axis_length - 2),
                    positive_mask.ndim,
                    output_axis,
                )
                negative_interior = _broadcast_axis_mask(
                    (center_indices >= 1) & (center_indices <= axis_length - 3),
                    negative_mask.ndim,
                    output_axis,
                )
                positive_mask &= positive_interior
                negative_mask &= negative_interior
            while positive_mask.ndim < q_partial.ndim:
                positive_mask = positive_mask[..., np.newaxis]
                negative_mask = negative_mask[..., np.newaxis]
            q_partial = np.where(positive_mask, positive_derivative, q_partial)
            q_partial = np.where(negative_mask, negative_derivative, q_partial)
        else:
            q_partial = _sampled_first_derivative(
                q_tensor_field,
                sample_indices,
                spatial_axes,
                component_index,
                spacing,
                periodic,
            )
        while velocity_component.ndim < q_partial.ndim:
            velocity_component = velocity_component[..., np.newaxis]
        result += velocity_component * q_partial
    return result


def _sampled_velocity_shear_and_rotation_tensors(
    velocity_field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int],
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray]:
    velocity_center = _take_spatial_points(velocity_field, sample_indices, spatial_axes)
    grad_v = np.zeros(velocity_center.shape + (3,), dtype=float)
    for derivative_axis_index, spacing in enumerate(spacings):
        grad_v[..., :, derivative_axis_index] = _sampled_first_derivative(
            velocity_field,
            sample_indices,
            spatial_axes,
            derivative_axis_index,
            spacing,
            periodic,
        )
    grad_v_t = np.swapaxes(grad_v, -1, -2)
    e_tensor = 0.5 * (grad_v + grad_v_t)
    omega_tensor = 0.5 * (grad_v - grad_v_t)
    return e_tensor, omega_tensor


def tensor_weak_inner_product(
    test_tensor: np.ndarray,
    target_tensor: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
) -> np.ndarray:
    """Compute the spatial integral of Phi : F for tensor-valued fields."""
    pointwise_inner = np.einsum("...ij,...ij->...", test_tensor, target_tensor)
    return integrate_over_space(
        pointwise_inner,
        spacings=spacings,
        spatial_axes=spatial_axes,
    )


def _symmetric_traceless_tensor_basis() -> tuple[np.ndarray, ...]:
    """Return a simple basis for symmetric traceless 3x3 tensors."""
    return (
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=float),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float),
    )


def build_local_symmetric_traceless_q_test_fields(
    shape: tuple[int, int, int],
    center_indices: tuple[int, int, int],
    half_widths: tuple[int, int, int],
    local_test_field_mode: str = "basic",
) -> tuple[np.ndarray, ...]:
    """Build compactly supported tensor test fields for a Q equation."""
    local_test_field_mode = _validate_local_test_field_mode(local_test_field_mode)
    bump = _build_local_bump(shape, center_indices, half_widths)

    x, y, z = np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(shape[1], dtype=float),
        np.arange(shape[2], dtype=float),
        indexing="ij",
    )
    cx, cy, cz = center_indices

    scalar_seeds = [bump]
    if local_test_field_mode in {"minimal", "basic", "expanded"}:
        scalar_seeds.extend(
            [
                bump * (x - cx),
                bump * (y - cy),
                bump * (z - cz),
            ]
        )
    if local_test_field_mode == "expanded":
        scalar_seeds.extend(
            [
                bump * (x - cx) ** 2,
                bump * (y - cy) ** 2,
                bump * (z - cz) ** 2,
                bump * (x - cx) * (y - cy),
                bump * (x - cx) * (z - cz),
                bump * (y - cy) * (z - cz),
            ]
        )

    basis_tensors = _symmetric_traceless_tensor_basis()
    test_fields: list[np.ndarray] = []
    for scalar_seed in scalar_seeds:
        for basis in basis_tensors:
            test_fields.append(scalar_seed[..., np.newaxis, np.newaxis] * basis)

    if local_test_field_mode == "single":
        return tuple(test_fields[:5])
    return tuple(test_fields)


def extract_independent_q_components(tensor_field: np.ndarray) -> np.ndarray:
    """Extract the five independent components of a symmetric traceless tensor."""
    tensor_field = np.asarray(tensor_field, dtype=float)
    if tensor_field.shape[-2:] != (3, 3):
        raise ValueError("Tensor field must have shape (..., 3, 3).")
    return np.stack(
        [
            tensor_field[..., 0, 0],
            tensor_field[..., 0, 1],
            tensor_field[..., 0, 2],
            tensor_field[..., 1, 1],
            tensor_field[..., 1, 2],
        ],
        axis=-1,
    )


def fit_be_q_equation_pointwise(
    velocity_field: np.ndarray,
    q_tensor_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2, 3),
    periodic: bool = False,
    spatial_derivative_method: str = "finite_difference",
    v_time_level: str | None = None,
    q_time_level: str = "current",
    time_discretization: str = "two_point",
    use_interior_time_only: bool = True,
    threshold: float = 1e-10,
    optimizer_alpha: float = 0.0,
    normalize_columns: bool = False,
    include_terms: Sequence[str] | None = None,
    sample_step: int | Sequence[int] | None = None,
    sample_slices: Sequence[object] | None = None,
    flow_alignment_lambda: float = 1.0,
    verbose: bool = False,
) -> BEQEquationPointwiseFitResult:
    """Fit a simplified Q equation pointwise using the five independent Q components.

    ``include_terms`` can restrict the fitted right-hand-side terms. Valid term names
    are ``material_q``, ``omega_q``, ``e_plain``, ``e_q``, ``q_colon_e_q``,
    ``flow_alignment``, ``bulk_linear``, ``bulk_quadratic``, ``bulk_cubic``,
    and ``elastic_l1``.

    ``sample_step`` and ``sample_slices`` control which spatial grid points enter
    the pointwise regression. Derivatives are still computed on the full input
    fields using ``spacings`` before any point samples are selected.

    ``spatial_derivative_method='upwind3'`` applies a third-order upwind stencil
    only to the advective ``material_q`` term. Other spatial derivatives in the
    same fit, such as ``Omega``/``E`` and ``Delta Q``, continue to use centered
    finite differences.

    ``v_time_level`` and ``q_time_level`` independently choose whether the
    velocity-dependent and Q-dependent right-hand-side factors are evaluated at
    ``"previous"``/``"n"`` or ``"current"``/``"n+1"``.

    ``time_discretization`` controls only the left-hand-side time derivative:
    ``"two_point"`` uses adjacent saved frames, and ``"central"`` uses
    ``(Q[n+1] - Q[n-1]) / (2 dt)``.
    """

    def emit_progress(message: str) -> None:
        if verbose:
            print(message, flush=True)

    spatial_axes = tuple(spatial_axes)
    spacings = tuple(float(value) for value in spacings)
    spatial_derivative_method = _normalize_q_advection_derivative_method(
        spatial_derivative_method
    )
    non_advection_derivative_method = _support_derivative_method_for_non_advection(
        spatial_derivative_method
    )
    time_discretization = _validate_time_discretization(time_discretization)
    v_time_level, q_time_level = _resolve_velocity_q_time_levels(
        v_time_level=v_time_level,
        q_time_level=q_time_level,
    )
    v_time_level = _validate_rhs_time_level("v_time_level", v_time_level)
    q_time_level = _validate_rhs_time_level("q_time_level", q_time_level)

    if q_tensor_field.shape[-2:] != (3, 3):
        raise ValueError("Q must have shape (..., 3, 3).")

    spatial_shape = tuple(q_tensor_field.shape[axis] for axis in spatial_axes)
    point_sample_slices = _normalize_pointwise_sample_slices(
        sample_slices,
        sample_step,
        spatial_shape,
    )
    included_feature_names = _validate_included_feature_names(
        include_terms,
        _Q_EQUATION_FEATURE_NAMES,
        _Q_EQUATION_DEFAULT_FEATURE_NAMES,
    )
    if velocity_field.shape[:-1] != q_tensor_field.shape[:-2]:
        raise ValueError(
            "Velocity and Q must share the same time/space layout; got "
            f"velocity shape {velocity_field.shape} and Q shape {q_tensor_field.shape}."
        )
    if velocity_field.shape[-1] != 3:
        raise ValueError("Velocity must have shape (..., 3).")
    if (
        point_sample_slices is not None
        and spatial_derivative_method not in {"finite_difference", "upwind3"}
    ):
        raise ValueError(
            "sample_step/sample_slices currently support only "
            "spatial_derivative_method='finite_difference' or 'upwind3'."
        )

    emit_progress(
        "Starting pointwise Q fit: "
        f"velocity shape={velocity_field.shape}, Q shape={q_tensor_field.shape}, "
        f"spatial_derivative_method={spatial_derivative_method}, "
        f"time_discretization={time_discretization}, "
        f"v_time_level={v_time_level}, "
        f"q_time_level={q_time_level}, "
        f"sample_slices={point_sample_slices}, "
        f"include_terms={included_feature_names}"
    )
    emit_progress("Computing dt(Q) and selected Q-equation terms...")
    (
        target_time_indices,
        velocity_time_indices,
        q_time_indices,
        effective_time_derivative_method,
    ) = (
        _time_indices_for_q_regression(
            q_tensor_field.shape[time_axis],
            use_interior_time_only=use_interior_time_only,
            time_discretization=time_discretization,
            v_time_level=v_time_level,
            q_time_level=q_time_level,
        )
    )

    dt_q = q_time_derivative(
        (
            q_tensor_field
            if point_sample_slices is None
            else _take_spatial_points(
                q_tensor_field,
                _pointwise_sample_indices(point_sample_slices, spatial_shape),
                spatial_axes,
            )
        ),
        dt=dt,
        time_axis=time_axis,
        periodic=False,
        time_derivative_method=effective_time_derivative_method,
    )
    raw_feature_by_name: dict[str, np.ndarray] = {}
    velocity_rhs = np.take(velocity_field, velocity_time_indices, axis=time_axis)
    q_rhs = np.take(q_tensor_field, q_time_indices, axis=time_axis)

    if point_sample_slices is None:
        q_center = q_rhs
        if "material_q" in included_feature_names:
            raw_feature_by_name["material_q"] = q_material_derivative(
                velocity_rhs,
                q_rhs,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=spatial_derivative_method,
            )
        if any(
            name in included_feature_names
            for name in ("omega_q", "e_plain", "e_q", "q_colon_e_q", "flow_alignment")
        ):
            e_tensor, omega_tensor = velocity_shear_and_rotation_tensors(
                velocity_rhs,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=non_advection_derivative_method,
            )
            if "omega_q" in included_feature_names:
                raw_feature_by_name["omega_q"] = q_omega_interaction(
                    q_rhs, omega_tensor
                )
            if "e_plain" in included_feature_names:
                raw_feature_by_name["e_plain"] = q_e_plain_interaction(e_tensor)
            if "e_q" in included_feature_names:
                raw_feature_by_name["e_q"] = q_e_interaction(q_rhs, e_tensor)
            if "q_colon_e_q" in included_feature_names:
                raw_feature_by_name["q_colon_e_q"] = q_colon_e_q_interaction(
                    q_rhs, e_tensor
                )
            if "flow_alignment" in included_feature_names:
                raw_feature_by_name["flow_alignment"] = q_flow_alignment(
                    q_rhs,
                    e_tensor,
                    omega_tensor,
                    lambda_flow_alignment=flow_alignment_lambda,
                )
        if "elastic_l1" in included_feature_names:
            raw_feature_by_name["elastic_l1"] = elastic_molecular_field_l1(
                q_rhs,
                l1=1.0,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=non_advection_derivative_method,
            )
    else:
        sample_indices = _pointwise_sample_indices(point_sample_slices, spatial_shape)
        q_center = _take_spatial_points(q_rhs, sample_indices, spatial_axes)
        if "material_q" in included_feature_names:
            raw_feature_by_name["material_q"] = _sampled_q_material_derivative(
                velocity_rhs,
                q_rhs,
                sample_indices,
                spacings,
                spatial_axes,
                periodic,
                spatial_derivative_method,
            )
        if any(
            name in included_feature_names
            for name in ("omega_q", "e_plain", "e_q", "q_colon_e_q", "flow_alignment")
        ):
            e_tensor, omega_tensor = _sampled_velocity_shear_and_rotation_tensors(
                velocity_rhs,
                sample_indices,
                spacings,
                spatial_axes,
                periodic,
            )
            if "omega_q" in included_feature_names:
                raw_feature_by_name["omega_q"] = q_omega_interaction(
                    q_center, omega_tensor
                )
            if "e_plain" in included_feature_names:
                raw_feature_by_name["e_plain"] = q_e_plain_interaction(e_tensor)
            if "e_q" in included_feature_names:
                raw_feature_by_name["e_q"] = q_e_interaction(q_center, e_tensor)
            if "q_colon_e_q" in included_feature_names:
                raw_feature_by_name["q_colon_e_q"] = q_colon_e_q_interaction(
                    q_center, e_tensor
                )
            if "flow_alignment" in included_feature_names:
                raw_feature_by_name["flow_alignment"] = q_flow_alignment(
                    q_center,
                    e_tensor,
                    omega_tensor,
                    lambda_flow_alignment=flow_alignment_lambda,
                )
        if "elastic_l1" in included_feature_names:
            raw_feature_by_name["elastic_l1"] = _sampled_tensor_laplacian(
                q_rhs,
                sample_indices,
                spacings,
                spatial_axes,
                periodic,
            )

    if "bulk_linear" in included_feature_names:
        raw_feature_by_name["bulk_linear"] = bulk_linear_term(q_center)
    if "bulk_quadratic" in included_feature_names:
        raw_feature_by_name["bulk_quadratic"] = bulk_quadratic_term(q_center)
    if "bulk_cubic" in included_feature_names:
        raw_feature_by_name["bulk_cubic"] = bulk_cubic_term(q_center)

    masked_target = np.take(
        extract_independent_q_components(dt_q),
        target_time_indices,
        axis=time_axis,
    )
    masked_target = _apply_pointwise_sample_slices(
        masked_target,
        None,
        spatial_axes,
    )

    emit_progress(
        "Building pointwise regression arrays from the five independent Q components..."
    )
    regression_feature_by_name = {}
    for feature_name in included_feature_names:
        masked_feature = extract_independent_q_components(
            raw_feature_by_name[feature_name]
        )
        regression_feature_by_name[feature_name] = _apply_pointwise_sample_slices(
            masked_feature,
            None,
            spatial_axes,
        )

    feature_names = included_feature_names
    regression_features = tuple(
        regression_feature_by_name[feature_name] for feature_name in feature_names
    )
    feature_array = np.stack(regression_features, axis=-1)

    emit_progress(
        "Starting sparse pointwise regression: "
        f"target shape={masked_target.shape}, feature shape={feature_array.shape}"
    )
    coefficients, prediction, r2, rmse, relative_residual = _fit_linear_term_model(
        masked_target,
        regression_features,
        threshold=threshold,
        alpha=optimizer_alpha,
        normalize_columns=normalize_columns,
    )
    emit_progress(
        "Finished pointwise Q fit: "
        f"r2={r2:.4f}, relative_residual={relative_residual:.4f}, "
        f"sample_count={int(masked_target.size)}"
    )
    coefficient_by_name = dict(zip(feature_names, coefficients))

    return BEQEquationPointwiseFitResult(
        dt_q_coefficient=1.0,
        material_q_coefficient=float(coefficient_by_name.get("material_q", 0.0)),
        omega_q_coefficient=float(coefficient_by_name.get("omega_q", 0.0)),
        e_plain_coefficient=float(coefficient_by_name.get("e_plain", 0.0)),
        e_q_coefficient=float(coefficient_by_name.get("e_q", 0.0)),
        q_colon_e_q_coefficient=float(
            coefficient_by_name.get("q_colon_e_q", 0.0)
        ),
        flow_alignment_coefficient=float(
            coefficient_by_name.get("flow_alignment", 0.0)
        ),
        bulk_linear_coefficient=float(coefficient_by_name.get("bulk_linear", 0.0)),
        bulk_quadratic_coefficient=float(
            coefficient_by_name.get("bulk_quadratic", 0.0)
        ),
        bulk_cubic_coefficient=float(coefficient_by_name.get("bulk_cubic", 0.0)),
        elastic_l1_coefficient=float(coefficient_by_name.get("elastic_l1", 0.0)),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(masked_target.size),
        pointwise_target=masked_target,
        pointwise_prediction=prediction.reshape(masked_target.shape),
        pointwise_features=feature_array,
        coefficients=coefficients,
        feature_names=feature_names,
        independent_component_names=("Qxx", "Qxy", "Qxz", "Qyy", "Qyz"),
        sample_slices=point_sample_slices,
    )


def fit_be_q_equation_local_weak_form(
    velocity_field: np.ndarray,
    q_tensor_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    patch_half_widths: tuple[int, int, int] = (3, 3, 3),
    patch_strides: tuple[int, int, int] = (4, 4, 4),
    patch_specs: Sequence[LocalPatchSpec] | None = None,
    include_domain_edge_patch: bool = False,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2, 3),
    periodic: bool = False,
    spatial_derivative_method: str = "finite_difference",
    v_time_level: str | None = None,
    q_time_level: str = "current",
    time_discretization: str = "two_point",
    local_test_field_mode: str = "basic",
    use_interior_time_only: bool = True,
    include_q_colon_e_q: bool = False,
    include_flow_alignment: bool = False,
    flow_alignment_lambda: float = 1.0,
    threshold: float = 1e-10,
    optimizer_alpha: float = 0.0,
    normalize_columns: bool = False,
    compute_features_locally: bool = False,
    verbose: bool = False,
    progress_interval: int = 10,
) -> BEQEquationLocalWeakFormFitResult:
    """Fit a simplified Q equation with local weak-form tensor test fields.

    The default fitted equation is

    ``dt(Q) = c_m (v . grad)Q + c_o (Omega Q - Q Omega) + c_e (E Q + Q E)
    + c_qe (Q:E)Q
    + c_b1 Q + c_b2 (Q^2 - I Tr(Q^2)/3) + c_b3 Tr(Q^2)Q
    + c_l1 Delta Q``

    with all right-hand-side terms entering with positive signs.
    Set ``include_q_colon_e_q=True`` to add the standalone nonlinear
    ``(Q:E)Q`` coupling term.
    Set ``include_flow_alignment=True`` to also include the full Beris-Edwards
    flow-alignment tensor ``S(W,Q)``.

    ``spatial_derivative_method='upwind3'`` applies a third-order upwind stencil
    only to the advective ``material_q`` term. Other spatial derivatives in the
    local weak-form pipeline continue to use centered finite differences.

    ``v_time_level`` and ``q_time_level`` independently choose whether the
    velocity-dependent and Q-dependent right-hand-side factors are evaluated at
    ``"previous"``/``"n"`` or ``"current"``/``"n+1"``.

    ``time_discretization`` controls only the left-hand-side time derivative:
    ``"two_point"`` uses adjacent saved frames, and ``"central"`` uses
    ``(Q[n+1] - Q[n-1]) / (2 dt)``.

    ``compute_features_locally=True`` avoids constructing the full-domain weak-form
    feature arrays before patch integration. Instead, each patch is cropped with a
    small halo, the strong-form features are evaluated on that local block, and only
    then are the weak integrals formed. This is especially useful for large 3D runs
    with ``spatial_derivative_method='upwind3'``.
    """

    def emit_progress(message: str) -> None:
        if verbose:
            print(message, flush=True)

    overall_start = time.perf_counter()
    spatial_axes = tuple(spatial_axes)
    spacings = tuple(float(value) for value in spacings)
    spatial_derivative_method = _normalize_q_advection_derivative_method(
        spatial_derivative_method
    )
    non_advection_derivative_method = _support_derivative_method_for_non_advection(
        spatial_derivative_method
    )
    time_discretization = _validate_time_discretization(time_discretization)
    v_time_level, q_time_level = _resolve_velocity_q_time_levels(
        v_time_level=v_time_level,
        q_time_level=q_time_level,
    )
    v_time_level = _validate_rhs_time_level("v_time_level", v_time_level)
    q_time_level = _validate_rhs_time_level("q_time_level", q_time_level)
    local_test_field_mode = _validate_local_test_field_mode(local_test_field_mode)

    if velocity_field.shape[:-1] != q_tensor_field.shape[:-2]:
        raise ValueError(
            "Velocity and Q must share the same time/space layout; got "
            f"velocity shape {velocity_field.shape} and Q shape {q_tensor_field.shape}."
        )
    if velocity_field.shape[-1] != 3:
        raise ValueError("Velocity must have shape (..., 3).")
    if q_tensor_field.shape[-2:] != (3, 3):
        raise ValueError("Q must have shape (..., 3, 3).")

    emit_progress(
        "Starting local weak-form Q fit: "
        f"velocity shape={velocity_field.shape}, Q shape={q_tensor_field.shape}, "
        f"spatial_derivative_method={spatial_derivative_method}, "
        f"time_discretization={time_discretization}, "
        f"v_time_level={v_time_level}, "
        f"q_time_level={q_time_level}, "
        f"local_test_field_mode={local_test_field_mode}"
    )
    extra_messages = []
    if include_q_colon_e_q:
        extra_messages.append("Q:E coupling")
    if include_flow_alignment:
        extra_messages.append("flow alignment")
    flow_message = ""
    if extra_messages:
        flow_message = ", " + ", ".join(extra_messages)
    emit_progress(
        "Computing dt(Q), material(Q), Omega/Q, E/Q"
        f"{flow_message}, bulk terms, and L1 term..."
    )
    (
        target_time_indices,
        velocity_time_indices,
        q_time_indices,
        effective_time_derivative_method,
    ) = (
        _time_indices_for_q_regression(
            q_tensor_field.shape[time_axis],
            use_interior_time_only=use_interior_time_only,
            time_discretization=time_discretization,
            v_time_level=v_time_level,
            q_time_level=q_time_level,
        )
    )

    dt_q = None
    dt_q_target = None
    velocity_rhs = None
    q_rhs = None
    material_q = None
    e_tensor = None
    omega_tensor = None
    omega_q = None
    e_q = None
    q_colon_e_q = None
    flow_alignment = None
    bulk_linear = None
    bulk_quadratic = None
    bulk_cubic = None
    elastic_l1 = None
    halo_widths = (
        (2, 2, 2) if spatial_derivative_method == "upwind3" else (1, 1, 1)
    )
    if not compute_features_locally:
        dt_q = q_time_derivative(
            q_tensor_field,
            dt=dt,
            time_axis=time_axis,
            periodic=False,
            time_derivative_method=effective_time_derivative_method,
        )
        dt_q_target = np.take(dt_q, target_time_indices, axis=time_axis)
        velocity_rhs = np.take(velocity_field, velocity_time_indices, axis=time_axis)
        q_rhs = np.take(q_tensor_field, q_time_indices, axis=time_axis)

        material_q = q_material_derivative(
            velocity_rhs,
            q_rhs,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
        e_tensor, omega_tensor = velocity_shear_and_rotation_tensors(
            velocity_rhs,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=non_advection_derivative_method,
        )
        omega_q = q_omega_interaction(q_rhs, omega_tensor)
        e_q = q_e_interaction(q_rhs, e_tensor)
        if include_q_colon_e_q:
            q_colon_e_q = q_colon_e_q_interaction(q_rhs, e_tensor)
        if include_flow_alignment:
            flow_alignment = q_flow_alignment(
                q_rhs,
                e_tensor,
                omega_tensor,
                lambda_flow_alignment=flow_alignment_lambda,
            )
        bulk_linear = bulk_linear_term(q_rhs)
        bulk_quadratic = bulk_quadratic_term(q_rhs)
        bulk_cubic = bulk_cubic_term(q_rhs)
        elastic_l1 = elastic_molecular_field_l1(
            q_rhs,
            l1=1.0,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=non_advection_derivative_method,
        )

    spatial_shape = tuple(q_tensor_field.shape[axis] for axis in spatial_axes)
    if patch_specs is None:
        patch_specs = generate_local_patch_specs(
            shape=spatial_shape,
            half_widths=patch_half_widths,
            strides=patch_strides,
            include_domain_edge_patch=include_domain_edge_patch,
        )
    else:
        patch_specs = tuple(patch_specs)

    emit_progress(
        "Starting local tensor patch integration: "
        f"{len(patch_specs)} patches, half_widths={patch_half_widths}, strides={patch_strides}"
    )

    local_targets: list[np.ndarray] = []
    local_material_q: list[np.ndarray] = []
    local_omega_q: list[np.ndarray] = []
    local_e_q: list[np.ndarray] = []
    local_q_colon_e_q: list[np.ndarray] = []
    local_flow_alignment: list[np.ndarray] = []
    local_bulk_linear: list[np.ndarray] = []
    local_bulk_quadratic: list[np.ndarray] = []
    local_bulk_cubic: list[np.ndarray] = []
    local_elastic_l1: list[np.ndarray] = []

    patch_loop_start = time.perf_counter()
    for patch_index, patch in enumerate(patch_specs, start=1):
        if compute_features_locally:
            local_velocity_field, local_patch = _crop_patch_with_halo(
                velocity_field,
                spatial_axes=spatial_axes,
                patch=patch,
                halo_widths=halo_widths,
                periodic=periodic,
            )
            local_q_tensor_field, local_patch = _crop_patch_with_halo(
                q_tensor_field,
                spatial_axes=spatial_axes,
                patch=patch,
                halo_widths=halo_widths,
                periodic=periodic,
            )
            local_spatial_shape = tuple(
                local_q_tensor_field.shape[axis] for axis in spatial_axes
            )
            local_dt_q = q_time_derivative(
                local_q_tensor_field,
                dt=dt,
                time_axis=time_axis,
                periodic=False,
                time_derivative_method=effective_time_derivative_method,
            )
            local_dt_q_target = np.take(local_dt_q, target_time_indices, axis=time_axis)
            local_velocity_rhs = np.take(
                local_velocity_field, velocity_time_indices, axis=time_axis
            )
            local_q_rhs = np.take(local_q_tensor_field, q_time_indices, axis=time_axis)
            local_material_q = q_material_derivative(
                local_velocity_rhs,
                local_q_rhs,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=spatial_derivative_method,
            )
            local_e_tensor, local_omega_tensor = velocity_shear_and_rotation_tensors(
                local_velocity_rhs,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=non_advection_derivative_method,
            )
            local_omega_q = q_omega_interaction(local_q_rhs, local_omega_tensor)
            local_e_q = q_e_interaction(local_q_rhs, local_e_tensor)
            local_q_colon_e_q_term = (
                q_colon_e_q_interaction(local_q_rhs, local_e_tensor)
                if include_q_colon_e_q
                else None
            )
            local_flow_alignment_term = (
                q_flow_alignment(
                    local_q_rhs,
                    local_e_tensor,
                    local_omega_tensor,
                    lambda_flow_alignment=flow_alignment_lambda,
                )
                if include_flow_alignment
                else None
            )
            local_bulk_linear_term = bulk_linear_term(local_q_rhs)
            local_bulk_quadratic_term = bulk_quadratic_term(local_q_rhs)
            local_bulk_cubic_term = bulk_cubic_term(local_q_rhs)
            local_elastic_l1_term = elastic_molecular_field_l1(
                local_q_rhs,
                l1=1.0,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=non_advection_derivative_method,
            )
        else:
            local_patch = patch
            local_spatial_shape = spatial_shape
            local_dt_q_target = dt_q_target
            local_material_q = material_q
            local_omega_q = omega_q
            local_e_q = e_q
            local_q_colon_e_q_term = q_colon_e_q
            local_flow_alignment_term = flow_alignment
            local_bulk_linear_term = bulk_linear
            local_bulk_quadratic_term = bulk_quadratic
            local_bulk_cubic_term = bulk_cubic
            local_elastic_l1_term = elastic_l1

        test_fields = build_local_symmetric_traceless_q_test_fields(
            shape=local_spatial_shape,
            center_indices=local_patch.center_indices,
            half_widths=local_patch.half_widths,
            local_test_field_mode=local_test_field_mode,
        )
        for test_field_spatial in test_fields:
            test_field = np.broadcast_to(
                test_field_spatial[np.newaxis, ...],
                local_dt_q_target.shape,
            )
            local_targets.append(
                tensor_weak_inner_product(
                    test_field,
                    local_dt_q_target,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_material_q.append(
                tensor_weak_inner_product(
                    test_field,
                    local_material_q,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_omega_q.append(
                tensor_weak_inner_product(
                    test_field,
                    local_omega_q,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_e_q.append(
                tensor_weak_inner_product(
                    test_field,
                    local_e_q,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            if local_q_colon_e_q_term is not None:
                local_q_colon_e_q.append(
                    tensor_weak_inner_product(
                        test_field,
                        local_q_colon_e_q_term,
                        spacings=spacings,
                        spatial_axes=spatial_axes,
                    )
                )
            if local_flow_alignment_term is not None:
                local_flow_alignment.append(
                    tensor_weak_inner_product(
                        test_field,
                        local_flow_alignment_term,
                        spacings=spacings,
                        spatial_axes=spatial_axes,
                    )
                )
            local_bulk_linear.append(
                tensor_weak_inner_product(
                    test_field,
                    local_bulk_linear_term,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_bulk_quadratic.append(
                tensor_weak_inner_product(
                    test_field,
                    local_bulk_quadratic_term,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_bulk_cubic.append(
                tensor_weak_inner_product(
                    test_field,
                    local_bulk_cubic_term,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_elastic_l1.append(
                tensor_weak_inner_product(
                    test_field,
                    local_elastic_l1_term,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )

        should_report = (
            patch_index == 1
            or patch_index == len(patch_specs)
            or patch_index % max(progress_interval, 1) == 0
        )
        if should_report:
            elapsed = time.perf_counter() - patch_loop_start
            average_per_patch = elapsed / patch_index
            remaining = max(len(patch_specs) - patch_index, 0) * average_per_patch
            emit_progress(
                "Completed tensor patch "
                f"{patch_index}/{len(patch_specs)} "
                f"(elapsed={elapsed:.1f}s, eta={remaining:.1f}s)"
            )

    emit_progress("Building regression arrays from local tensor weak-form samples...")
    local_target = np.stack(local_targets, axis=0)
    local_material_q_feature = np.stack(local_material_q, axis=0)
    local_omega_q_feature = np.stack(local_omega_q, axis=0)
    local_e_q_feature = np.stack(local_e_q, axis=0)
    local_q_colon_e_q_feature = (
        np.stack(local_q_colon_e_q, axis=0) if include_q_colon_e_q else None
    )
    local_flow_alignment_feature = (
        np.stack(local_flow_alignment, axis=0) if include_flow_alignment else None
    )
    local_bulk_linear_feature = np.stack(local_bulk_linear, axis=0)
    local_bulk_quadratic_feature = np.stack(local_bulk_quadratic, axis=0)
    local_bulk_cubic_feature = np.stack(local_bulk_cubic, axis=0)
    local_elastic_l1_feature = np.stack(local_elastic_l1, axis=0)

    masked_target = local_target
    masked_material_q = local_material_q_feature
    masked_omega_q = local_omega_q_feature
    masked_e_q = local_e_q_feature
    masked_q_colon_e_q = local_q_colon_e_q_feature
    masked_flow_alignment = local_flow_alignment_feature
    masked_bulk_linear = local_bulk_linear_feature
    masked_bulk_quadratic = local_bulk_quadratic_feature
    masked_bulk_cubic = local_bulk_cubic_feature
    masked_elastic_l1 = local_elastic_l1_feature

    feature_names = [
        "local_material_q",
        "local_omega_q",
        "local_e_q",
        "local_bulk_linear",
        "local_bulk_quadratic",
        "local_bulk_cubic",
        "local_elastic_l1",
    ]
    regression_features = [
        masked_material_q,
        masked_omega_q,
        masked_e_q,
        masked_bulk_linear,
        masked_bulk_quadratic,
        masked_bulk_cubic,
        masked_elastic_l1,
    ]
    if masked_q_colon_e_q is not None:
        feature_names.insert(3, "local_q_colon_e_q")
        regression_features.insert(3, masked_q_colon_e_q)
    if masked_flow_alignment is not None:
        insert_index = 4 if masked_q_colon_e_q is not None else 3
        feature_names.insert(insert_index, "local_flow_alignment")
        regression_features.insert(insert_index, masked_flow_alignment)
    feature_names = tuple(feature_names)
    regression_features = tuple(regression_features)
    feature_array = np.stack(regression_features, axis=-1)

    emit_progress(
        "Starting sparse tensor regression: "
        f"target shape={masked_target.shape}, feature shape={feature_array.shape}"
    )
    coefficients, prediction, r2, rmse, relative_residual = _fit_linear_term_model(
        masked_target,
        regression_features,
        threshold=threshold,
        alpha=optimizer_alpha,
        normalize_columns=normalize_columns,
    )
    emit_progress(
        "Finished local weak-form Q fit: "
        f"elapsed={time.perf_counter() - overall_start:.1f}s, "
        f"r2={r2:.4f}, relative_residual={relative_residual:.4f}"
    )

    coefficient_by_name = dict(zip(feature_names, coefficients))

    return BEQEquationLocalWeakFormFitResult(
        dt_q_coefficient=1.0,
        material_q_coefficient=float(coefficient_by_name.get("local_material_q", 0.0)),
        omega_q_coefficient=float(coefficient_by_name.get("local_omega_q", 0.0)),
        e_plain_coefficient=0.0,
        e_q_coefficient=float(coefficient_by_name.get("local_e_q", 0.0)),
        q_colon_e_q_coefficient=float(
            coefficient_by_name.get("local_q_colon_e_q", 0.0)
        ),
        flow_alignment_coefficient=float(
            coefficient_by_name.get("local_flow_alignment", 0.0)
        ),
        bulk_linear_coefficient=float(coefficient_by_name.get("local_bulk_linear", 0.0)),
        bulk_quadratic_coefficient=float(
            coefficient_by_name.get("local_bulk_quadratic", 0.0)
        ),
        bulk_cubic_coefficient=float(coefficient_by_name.get("local_bulk_cubic", 0.0)),
        elastic_l1_coefficient=float(coefficient_by_name.get("local_elastic_l1", 0.0)),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(masked_target.size),
        local_target=masked_target,
        local_prediction=prediction.reshape(masked_target.shape),
        local_features=feature_array,
        coefficients=coefficients,
        feature_names=feature_names,
        patch_specs=tuple(patch_specs),
        target_name="local_dt_q",
    )
