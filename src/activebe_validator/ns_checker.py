"""Navier-Stokes coefficient checks from velocity-only data."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ._regression import fit_linear_term_model as _fit_linear_term_model
from .operators import (
    central_difference,
    tensor_divergence,
    vector_laplacian,
    vector_material_derivative,
    vector_time_derivative,
)
from .velocity_terms import (
    TestFieldSpec,
    active_q_divergence_force,
    aligned_quadratic_velocity,
    build_beltrami_test_field,
    passive_backflow_force,
)


@dataclass(frozen=True)
class LocalPatchSpec:
    """Description of one local weak-form patch."""

    center_indices: tuple[int, int, int]
    half_widths: tuple[int, int, int]


def _crop_patch_with_halo(
    field: np.ndarray,
    *,
    spatial_axes: Sequence[int],
    patch: LocalPatchSpec,
    halo_widths: tuple[int, int, int],
    periodic: bool,
) -> tuple[np.ndarray, LocalPatchSpec]:
    """Crop one spatial patch with halo cells, preserving all non-spatial axes.

    The returned patch spec is expressed in the cropped-array coordinates so it
    can be passed directly to the local weak-form test-field builders.
    """
    cropped = np.asarray(field)
    local_centers: list[int] = []

    for derivative_axis_index, axis in enumerate(spatial_axes):
        axis_length = field.shape[axis]
        center = patch.center_indices[derivative_axis_index]
        half_width = patch.half_widths[derivative_axis_index]
        halo = halo_widths[derivative_axis_index]

        start = center - half_width - halo
        stop = center + half_width + halo + 1
        raw_indices = np.arange(start, stop, dtype=int)
        if periodic:
            axis_indices = raw_indices % axis_length
        else:
            axis_indices = np.clip(raw_indices, 0, axis_length - 1)

        cropped = np.take(cropped, axis_indices, axis=axis)
        center_matches = np.where(axis_indices == center)[0]
        if center_matches.size == 0:
            raise ValueError(
                "Failed to locate the patch center inside the cropped indices; "
                f"axis={axis}, center={center}, indices={axis_indices}."
            )
        local_centers.append(int(center_matches[0]))

    local_patch = LocalPatchSpec(
        center_indices=tuple(local_centers),
        half_widths=patch.half_widths,
    )
    return cropped, local_patch


@dataclass(frozen=True)
class NSWeakFormFitResult:
    """Summary of a weak-form SINDy-style Navier-Stokes coefficient fit."""

    material_coefficient: float
    viscosity_coefficient: float
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int
    weak_target: np.ndarray
    weak_prediction: np.ndarray
    weak_features: np.ndarray
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    test_field_specs: tuple[TestFieldSpec, ...]


@dataclass(frozen=True)
class NSProjectedFitResult:
    """Summary of a pressure-free projected Navier-Stokes coefficient fit."""

    material_coefficient: float
    viscosity_coefficient: float
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int
    projected_target: np.ndarray
    projected_prediction: np.ndarray
    projected_features: np.ndarray
    coefficients: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class NSPointwiseFitResult:
    """Summary of a strong-form pointwise Navier-Stokes coefficient fit."""

    dt_coefficient: float
    material_coefficient: float
    viscosity_coefficient: float
    pressure_gradient_coefficient: float
    linear_velocity_coefficient: float
    quadratic_velocity_coefficient: float
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int
    pointwise_target: np.ndarray
    pointwise_prediction: np.ndarray
    pointwise_features: np.ndarray
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    sample_slices: tuple[object, object, object] | None = None


@dataclass(frozen=True)
class NSLocalWeakFormFitResult:
    """Summary of a local weak-form Navier-Stokes coefficient fit."""

    material_coefficient: float
    viscosity_coefficient: float
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


@dataclass(frozen=True)
class BEEquationVelocityFitResult:
    """Summary of a weak-form velocity fit for the BE-style momentum equation."""

    material_coefficient: float
    viscosity_coefficient: float
    q_divergence_coefficient: float
    linear_velocity_coefficient: float
    quadratic_velocity_coefficient: float
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int
    weak_target: np.ndarray
    weak_prediction: np.ndarray
    weak_features: np.ndarray
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    test_field_specs: tuple[TestFieldSpec, ...]


@dataclass(frozen=True)
class BELocalEquationVelocityFitResult:
    """Summary of a local weak-form velocity fit for the BE-style equation."""

    dt_coefficient: float
    material_coefficient: float
    viscosity_coefficient: float
    q_divergence_coefficient: float
    passive_backflow_coefficient: float
    linear_velocity_coefficient: float
    quadratic_velocity_coefficient: float
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int


@dataclass(frozen=True)
class BEPointwiseVelocityFitResult:
    """Summary of a strong-form pointwise velocity fit for the BE-style equation."""

    dt_coefficient: float
    material_coefficient: float
    viscosity_coefficient: float
    pressure_gradient_coefficient: float
    q_divergence_coefficient: float
    passive_backflow_coefficient: float
    linear_velocity_coefficient: float
    quadratic_velocity_coefficient: float
    r2: float
    rmse: float
    relative_residual: float
    sample_count: int
    pointwise_target: np.ndarray
    pointwise_prediction: np.ndarray
    pointwise_features: np.ndarray
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    sample_slices: tuple[object, object, object] | None = None


DEFAULT_TEST_FIELD_SPECS: tuple[TestFieldSpec, ...] = (
    TestFieldSpec(a=1.0, b=0.3, c=0.7, wave_number=1),
    TestFieldSpec(a=0.4, b=1.1, c=0.5, wave_number=1),
    TestFieldSpec(a=0.8, b=0.6, c=1.2, wave_number=1),
)

_NS_POINTWISE_FEATURE_NAMES = (
    "linear_velocity",
    "quadratic_velocity",
    "material",
    "laplacian",
    "pressure_gradient_over_density",
)
_NS_POINTWISE_DEFAULT_FEATURE_NAMES = (
    "material",
    "laplacian",
    "pressure_gradient_over_density",
)
_BE_POINTWISE_FEATURE_NAMES = (
    "linear_velocity",
    "quadratic_velocity",
    "material",
    "laplacian",
    "pressure_gradient_over_density",
    "q_divergence",
    "passive_backflow",
)
_BE_POINTWISE_DEFAULT_FEATURE_NAMES = (
    "material",
    "laplacian",
    "pressure_gradient_over_density",
    "q_divergence",
    "passive_backflow",
)


def _validate_local_test_field_mode(mode: str) -> str:
    mode = mode.lower()
    if mode not in {"single", "minimal", "basic", "expanded"}:
        raise ValueError(
            "local_test_field_mode must be one of 'single', 'minimal', "
            f"'basic', or 'expanded', got {mode!r}."
        )
    return mode


def _validate_num_test_fields_per_patch(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("num_test_fields_per_patch must be an integer.")
    value = int(value)
    if value not in {1, 3, 12, 30}:
        raise ValueError(
            "num_test_fields_per_patch must be one of 1, 3, 12, or 30; "
            f"got {value}."
        )
    return value


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

def integrate_over_space(
    scalar_field: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Iterable[int],
) -> np.ndarray:
    """Integrate a scalar field over the spatial axes, preserving others."""
    spatial_axes = tuple(sorted(spatial_axes))
    volume_element = float(np.prod(tuple(spacings)))
    return np.sum(scalar_field, axis=spatial_axes) * volume_element


def weak_inner_product(
    test_field: np.ndarray,
    target_field: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Iterable[int],
) -> np.ndarray:
    """Compute the spatial integral of phi dot f, preserving non-spatial axes."""
    pointwise_dot = np.sum(test_field * target_field, axis=-1)
    return integrate_over_space(
        pointwise_dot,
        spacings=spacings,
        spatial_axes=spatial_axes,
    )

def _default_coordinates(num_points: int, spacing: float) -> np.ndarray:
    return np.arange(num_points, dtype=float) * spacing


def _spatial_grid_from_field(
    velocity_field: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        _default_coordinates(velocity_field.shape[axis], spacing)
        for spacing, axis in zip(spacings, spatial_axes)
    )


def _prepare_time_slice_mask(
    count: int,
    use_interior_time_only: bool,
) -> np.ndarray:
    mask = np.ones(count, dtype=bool)
    if use_interior_time_only and count > 2:
        mask[[0, -1]] = False
    return mask


def _fourier_wavenumbers(count: int) -> np.ndarray:
    return np.fft.fftfreq(count, d=1.0 / count)


def _spectral_scalar_first_derivative_periodic(
    field: np.ndarray,
    spacing: float,
    axis: int,
) -> np.ndarray:
    count = field.shape[axis]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    reshape = [1] * field.ndim
    reshape[axis] = count
    multiplier = (1.0j * wave_numbers).reshape(reshape)

    field_hat = np.fft.fft(field, axis=axis)
    derivative_hat = multiplier * field_hat
    return np.fft.ifft(derivative_hat, axis=axis).real


def _scalar_gradient(
    scalar_field: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
    *,
    periodic: bool,
    spatial_derivative_method: str,
) -> np.ndarray:
    method = spatial_derivative_method.lower()
    if method not in {"finite_difference", "spectral"}:
        raise ValueError(
            "spatial_derivative_method must be either 'finite_difference' or "
            f"'spectral', got {spatial_derivative_method!r}."
        )

    components: list[np.ndarray] = []
    for spacing, axis in zip(spacings, spatial_axes):
        if method == "spectral":
            if not periodic:
                raise ValueError(
                    "The spectral spatial derivative method requires periodic=True."
                )
            components.append(
                _spectral_scalar_first_derivative_periodic(
                    scalar_field,
                    spacing=spacing,
                    axis=axis,
                )
            )
        else:
            components.append(
                central_difference(
                    scalar_field,
                    spacing=spacing,
                    axis=axis,
                    periodic=periodic,
                )
            )
    return np.stack(components, axis=-1)


def _normalize_density_field(
    density_field: np.ndarray | float,
    reference_shape: tuple[int, ...],
) -> np.ndarray:
    density_array = np.asarray(density_field, dtype=float)
    if density_array.ndim == 0:
        density_array = np.full(reference_shape, float(density_array), dtype=float)
    elif density_array.shape != reference_shape:
        raise ValueError(
            "density_field must be a scalar or have the same time/space shape as "
            f"pressure_field; got density shape {density_array.shape} and expected {reference_shape}."
        )

    if np.any(density_array == 0.0):
        raise ValueError("density_field must be nonzero everywhere.")
    return density_array


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
        normalized_axis = axis % sampled_field.ndim
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


def _sampled_vector_laplacian(
    vector_field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int],
    periodic: bool,
) -> np.ndarray:
    result = np.zeros_like(
        _take_spatial_points(vector_field, sample_indices, spatial_axes)
    )
    for derivative_axis_index, spacing in enumerate(spacings):
        result += _sampled_second_derivative(
            vector_field,
            sample_indices,
            spatial_axes,
            derivative_axis_index,
            spacing,
            periodic,
        )
    return result


def _sampled_scalar_gradient(
    scalar_field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int],
    periodic: bool,
) -> np.ndarray:
    components = []
    for derivative_axis_index, spacing in enumerate(spacings):
        components.append(
            _sampled_first_derivative(
                scalar_field,
                sample_indices,
                spatial_axes,
                derivative_axis_index,
                spacing,
                periodic,
            )
        )
    return np.stack(components, axis=-1)


def _sampled_vector_material_derivative(
    vector_field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int],
    periodic: bool,
) -> np.ndarray:
    velocity_center = _take_spatial_points(vector_field, sample_indices, spatial_axes)
    result = np.zeros_like(velocity_center)
    for component_index, spacing in enumerate(spacings):
        partial = _sampled_first_derivative(
            vector_field,
            sample_indices,
            spatial_axes,
            component_index,
            spacing,
            periodic,
        )
        velocity_component = velocity_center[..., component_index]
        while velocity_component.ndim < partial.ndim:
            velocity_component = velocity_component[..., np.newaxis]
        result += velocity_component * partial
    return result


def _sampled_tensor_divergence(
    tensor_field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int],
    periodic: bool,
) -> np.ndarray:
    divergence = np.zeros_like(
        _take_spatial_points(tensor_field, sample_indices, spatial_axes)[..., 0]
    )
    for derivative_axis_index, spacing in enumerate(spacings):
        partial = _sampled_first_derivative(
            tensor_field,
            sample_indices,
            spatial_axes,
            derivative_axis_index,
            spacing,
            periodic,
        )
        divergence += partial[..., :, derivative_axis_index]
    return divergence


def leray_project_periodic_field(
    vector_field: np.ndarray,
    spatial_axes: Sequence[int] = (1, 2, 3),
) -> np.ndarray:
    """Project a periodic vector field onto its divergence-free component."""
    spatial_axes = tuple(spatial_axes)

    canonical = np.moveaxis(
        vector_field,
        (*spatial_axes, vector_field.ndim - 1),
        (1, 2, 3, 4),
    )
    nx, ny, nz = canonical.shape[1:4]
    kx = _fourier_wavenumbers(nx)
    ky = _fourier_wavenumbers(ny)
    kz = _fourier_wavenumbers(nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX**2 + KY**2 + KZ**2

    field_hat = np.fft.fftn(canonical, axes=(1, 2, 3))
    dot = (
        KX[None, ...] * field_hat[..., 0]
        + KY[None, ...] * field_hat[..., 1]
        + KZ[None, ...] * field_hat[..., 2]
    )
    mask = K2 > 0.0
    safe_k2 = np.where(mask, K2, 1.0)

    projected_hat = field_hat.copy()
    projected_hat[..., 0] -= np.where(mask, KX[None, ...] * dot / safe_k2[None, ...], 0.0)
    projected_hat[..., 1] -= np.where(mask, KY[None, ...] * dot / safe_k2[None, ...], 0.0)
    projected_hat[..., 2] -= np.where(mask, KZ[None, ...] * dot / safe_k2[None, ...], 0.0)
    projected_hat[:, ~mask, :] = 0.0

    projected = np.fft.ifftn(projected_hat, axes=(1, 2, 3)).real
    return np.moveaxis(projected, (1, 2, 3, 4), (*spatial_axes, vector_field.ndim - 1))


def _build_local_bump(
    shape: tuple[int, int, int],
    center_indices: tuple[int, int, int],
    half_widths: tuple[int, int, int],
) -> np.ndarray:
    """Build a smooth compact-support bump centered on one patch."""
    coordinates = np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(shape[1], dtype=float),
        np.arange(shape[2], dtype=float),
        indexing="ij",
    )

    bump = np.ones(shape, dtype=float)
    for axis_coords, center, half_width in zip(coordinates, center_indices, half_widths):
        normalized = (axis_coords - center) / max(half_width, 1)
        axis_weight = np.where(
            np.abs(normalized) < 1.0,
            (1.0 - normalized**2) ** 2,
            0.0,
        )
        bump *= axis_weight
    return bump


def build_local_divergence_free_test_fields(
    shape: tuple[int, int, int],
    center_indices: tuple[int, int, int],
    half_widths: tuple[int, int, int],
    spacings: Sequence[float],
    num_test_fields_per_patch: int = 1,
) -> tuple[np.ndarray, ...]:
    """Build a small bank of compactly supported divergence-free test fields."""
    num_test_fields_per_patch = _validate_num_test_fields_per_patch(
        num_test_fields_per_patch
    )
    bump = _build_local_bump(shape, center_indices, half_widths)
    dx, dy, dz = spacings

    # Three vector-potential-inspired scalar seeds with different anisotropy.
    X, Y, Z = np.meshgrid(
        np.arange(shape[0], dtype=float),
        np.arange(shape[1], dtype=float),
        np.arange(shape[2], dtype=float),
        indexing="ij",
    )
    cx, cy, cz = center_indices

    seeds = [bump]
    if num_test_fields_per_patch >= 12:
        seeds.extend(
            [
                bump * (X - cx),
                bump * (Y - cy),
                bump * (Z - cz),
            ]
        )
    if num_test_fields_per_patch == 30:
        seeds.extend(
            [
                bump * (X - cx) ** 2,
                bump * (Y - cy) ** 2,
                bump * (Z - cz) ** 2,
                bump * (X - cx) * (Y - cy),
                bump * (X - cx) * (Z - cz),
                bump * (Y - cy) * (Z - cz),
            ]
        )

    test_fields: list[np.ndarray] = []
    for psi in seeds:
        test_fields.append(
            np.stack(
                [
                    central_difference(psi, dy, 1, periodic=False),
                    -central_difference(psi, dx, 0, periodic=False),
                    np.zeros_like(psi),
                ],
                axis=-1,
            )
        )
        test_fields.append(
            np.stack(
                [
                    -central_difference(psi, dz, 2, periodic=False),
                    np.zeros_like(psi),
                    central_difference(psi, dx, 0, periodic=False),
                ],
                axis=-1,
            )
        )
        test_fields.append(
            np.stack(
                [
                    np.zeros_like(psi),
                    central_difference(psi, dz, 2, periodic=False),
                    -central_difference(psi, dy, 1, periodic=False),
                ],
                axis=-1,
            )
        )
    if num_test_fields_per_patch == 1:
        return (test_fields[0],)
    return tuple(test_fields)


def generate_local_patch_specs(
    shape: tuple[int, int, int],
    half_widths: tuple[int, int, int],
    strides: tuple[int, int, int],
    include_domain_edge_patch: bool = False,
) -> tuple[LocalPatchSpec, ...]:
    """Generate regularly spaced local patches over a 3D grid."""
    patch_specs: list[LocalPatchSpec] = []

    def axis_centers(length: int, half_width: int, stride: int) -> list[int]:
        start = half_width
        stop = length - half_width
        if stop <= start:
            return [length // 2]
        centers = list(range(start, stop, max(stride, 1)))
        if include_domain_edge_patch and centers[-1] != stop - 1:
            centers.append(stop - 1)
        return centers

    centers_x = axis_centers(shape[0], half_widths[0], strides[0])
    centers_y = axis_centers(shape[1], half_widths[1], strides[1])
    centers_z = axis_centers(shape[2], half_widths[2], strides[2])

    for ix in centers_x:
        for iy in centers_y:
            for iz in centers_z:
                patch_specs.append(
                    LocalPatchSpec(
                        center_indices=(ix, iy, iz),
                        half_widths=half_widths,
                    )
                )
    return tuple(patch_specs)


def fit_ns_weak_form_coefficients(
    velocity_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    spatial_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    test_fields: Sequence[np.ndarray] | None = None,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2, 3),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    time_derivative_method: str = "central",
    use_interior_time_only: bool = True,
    test_field_specs: Sequence[TestFieldSpec] = DEFAULT_TEST_FIELD_SPECS,
    threshold: float = 1e-10,
    alpha: float = 0.0,
    normalize_columns: bool = False,
) -> NSWeakFormFitResult:
    """Fit material-advection and viscosity coefficients from velocity-only data.

    This is the original global weak-form variant, intended mainly for periodic
    settings with globally defined divergence-free test fields.
    """
    spatial_axes = tuple(spatial_axes)
    spacings = tuple(float(value) for value in spacings)
    test_field_specs = tuple(test_field_specs)

    if spatial_grid is None:
        spatial_grid = _spatial_grid_from_field(velocity_field, spacings, spatial_axes)

    dt_velocity = vector_time_derivative(
        velocity_field,
        dt=dt,
        time_axis=time_axis,
        periodic=False,
        time_derivative_method=time_derivative_method,
    )
    material_velocity = vector_material_derivative(
        velocity_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    laplacian_velocity = vector_laplacian(
        velocity_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )

    weak_targets: list[np.ndarray] = []
    weak_material: list[np.ndarray] = []
    weak_laplacian: list[np.ndarray] = []

    if test_fields is None:
        test_field_bank = [
            build_beltrami_test_field(*spatial_grid, spec) for spec in test_field_specs
        ]
    else:
        test_field_bank = [np.asarray(field, dtype=float) for field in test_fields]
        expected_shape = tuple(velocity_field.shape[axis] for axis in spatial_axes) + (3,)
        for field in test_field_bank:
            if field.shape != expected_shape:
                raise ValueError(
                    "Each custom test field must have shape "
                    f"{expected_shape}, got {field.shape}."
                )

    for test_field_spatial in test_field_bank:
        test_field = np.broadcast_to(
            test_field_spatial[np.newaxis, ...],
            velocity_field.shape,
        )
        weak_targets.append(
            weak_inner_product(
                test_field,
                dt_velocity,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
        )
        weak_material.append(
            weak_inner_product(
                test_field,
                material_velocity,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
        )
        weak_laplacian.append(
            weak_inner_product(
                test_field,
                laplacian_velocity,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
        )

    weak_target = np.stack(weak_targets, axis=0)
    weak_material_feature = np.stack(weak_material, axis=0)
    weak_laplacian_feature = np.stack(weak_laplacian, axis=0)

    time_mask = _prepare_time_slice_mask(
        weak_target.shape[1],
        use_interior_time_only=use_interior_time_only,
    )
    masked_target = weak_target[:, time_mask]
    masked_material = weak_material_feature[:, time_mask]
    masked_laplacian = weak_laplacian_feature[:, time_mask]

    coefficients, prediction, r2, rmse, relative_residual = _fit_linear_term_model(
        masked_target,
        (masked_material, masked_laplacian),
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
    )

    return NSWeakFormFitResult(
        material_coefficient=float(coefficients[0]),
        viscosity_coefficient=float(coefficients[1]),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(masked_target.size),
        weak_target=masked_target,
        weak_prediction=prediction.reshape(masked_target.shape),
        weak_features=np.stack([masked_material, masked_laplacian], axis=-1),
        coefficients=coefficients,
        feature_names=("material", "laplacian"),
        test_field_specs=test_field_specs if test_fields is None else (),
    )


def fit_ns_local_weak_form_coefficients(
    velocity_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    patch_half_widths: tuple[int, int, int] = (3, 3, 3),
    patch_strides: tuple[int, int, int] = (4, 4, 4),
    patch_specs: Sequence[LocalPatchSpec] | None = None,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2, 3),
    periodic: bool = False,
    spatial_derivative_method: str = "finite_difference",
    time_derivative_method: str = "central",
    num_test_fields_per_patch: int = 1,
    use_interior_time_only: bool = True,
    threshold: float = 1e-10,
    alpha: float = 0.0,
    normalize_columns: bool = False,
) -> NSLocalWeakFormFitResult:
    """Fit NS coefficients with a local weak form built from compact patches.

    Each sample is a patch-wise weak integral, not a pointwise residual.
    The patch test functions are compactly supported and divergence-free, making
    this formulation suitable for non-periodic settings as well.
    """
    spatial_axes = tuple(spatial_axes)
    spacings = tuple(float(value) for value in spacings)
    num_test_fields_per_patch = _validate_num_test_fields_per_patch(
        num_test_fields_per_patch
    )

    dt_velocity = vector_time_derivative(
        velocity_field,
        dt=dt,
        time_axis=time_axis,
        periodic=False,
        time_derivative_method=time_derivative_method,
    )
    material_velocity = vector_material_derivative(
        velocity_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    laplacian_velocity = vector_laplacian(
        velocity_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )

    spatial_shape = tuple(velocity_field.shape[axis] for axis in spatial_axes)
    if patch_specs is None:
        patch_specs = generate_local_patch_specs(
            shape=spatial_shape,
            half_widths=patch_half_widths,
            strides=patch_strides,
        )
    else:
        patch_specs = tuple(patch_specs)

    local_targets: list[np.ndarray] = []
    local_material: list[np.ndarray] = []
    local_laplacian: list[np.ndarray] = []

    for patch in patch_specs:
        test_fields = build_local_divergence_free_test_fields(
            shape=spatial_shape,
            center_indices=patch.center_indices,
            half_widths=patch.half_widths,
            spacings=spacings,
            num_test_fields_per_patch=num_test_fields_per_patch,
        )
        for test_field_spatial in test_fields:
            test_field = np.broadcast_to(
                test_field_spatial[np.newaxis, ...],
                velocity_field.shape,
            )
            local_targets.append(
                weak_inner_product(
                    test_field,
                    dt_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_material.append(
                weak_inner_product(
                    test_field,
                    material_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_laplacian.append(
                weak_inner_product(
                    test_field,
                    laplacian_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )

    local_target = np.stack(local_targets, axis=0)
    local_material_feature = np.stack(local_material, axis=0)
    local_laplacian_feature = np.stack(local_laplacian, axis=0)

    time_mask = _prepare_time_slice_mask(
        local_target.shape[1],
        use_interior_time_only=use_interior_time_only,
    )
    masked_target = local_target[:, time_mask]
    masked_material = local_material_feature[:, time_mask]
    masked_laplacian = local_laplacian_feature[:, time_mask]

    coefficients, prediction, r2, rmse, relative_residual = _fit_linear_term_model(
        masked_target,
        (masked_material, masked_laplacian),
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
    )

    return NSLocalWeakFormFitResult(
        material_coefficient=float(coefficients[0]),
        viscosity_coefficient=float(coefficients[1]),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(masked_target.size),
        local_target=masked_target,
        local_prediction=prediction.reshape(masked_target.shape),
        local_features=np.stack([masked_material, masked_laplacian], axis=-1),
        coefficients=coefficients,
        feature_names=("local_material", "local_laplacian"),
        patch_specs=tuple(patch_specs),
    )


def fit_ns_projected_coefficients(
    velocity_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2, 3),
    time_derivative_method: str = "central",
    use_interior_time_only: bool = True,
    threshold: float = 1e-10,
    alpha: float = 0.0,
    normalize_columns: bool = False,
) -> NSProjectedFitResult:
    """Fit NS coefficients from velocity-only data using Leray projection.

    This checker assumes a periodic box. It removes the pressure gradient by
    applying the Leray projection to the momentum equation.
    """
    spatial_axes = tuple(spatial_axes)
    spacings = tuple(float(value) for value in spacings)

    dt_velocity = vector_time_derivative(
        velocity_field,
        dt=dt,
        time_axis=time_axis,
        periodic=False,
        time_derivative_method=time_derivative_method,
    )
    material_velocity = vector_material_derivative(
        velocity_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=True,
    )
    laplacian_velocity = vector_laplacian(
        velocity_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=True,
    )

    projected_target = leray_project_periodic_field(
        dt_velocity,
        spatial_axes=spatial_axes,
    )
    projected_material = leray_project_periodic_field(
        material_velocity,
        spatial_axes=spatial_axes,
    )
    projected_laplacian = leray_project_periodic_field(
        laplacian_velocity,
        spatial_axes=spatial_axes,
    )

    time_mask = _prepare_time_slice_mask(
        projected_target.shape[time_axis],
        use_interior_time_only=use_interior_time_only,
    )
    masked_target = np.compress(time_mask, projected_target, axis=time_axis)
    masked_material = np.compress(time_mask, projected_material, axis=time_axis)
    masked_laplacian = np.compress(time_mask, projected_laplacian, axis=time_axis)

    coefficients, prediction, r2, rmse, relative_residual = _fit_linear_term_model(
        masked_target,
        (masked_material, masked_laplacian),
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
    )

    return NSProjectedFitResult(
        material_coefficient=float(coefficients[0]),
        viscosity_coefficient=float(coefficients[1]),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(masked_target.size),
        projected_target=masked_target,
        projected_prediction=prediction.reshape(masked_target.shape),
        projected_features=np.stack([masked_material, masked_laplacian], axis=-1),
        coefficients=coefficients,
        feature_names=("projected_material", "projected_laplacian"),
    )


def fit_ns_pointwise_coefficients(
    velocity_field: np.ndarray,
    pressure_field: np.ndarray,
    density_field: np.ndarray | float,
    dt: float,
    spacings: Sequence[float],
    *,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2, 3),
    periodic: bool = False,
    spatial_derivative_method: str = "finite_difference",
    time_derivative_method: str = "central",
    use_interior_time_only: bool = True,
    include_terms: Sequence[str] | None = None,
    sample_step: int | Sequence[int] | None = None,
    sample_slices: Sequence[object] | None = None,
    threshold: float = 1e-10,
    alpha: float = 0.0,
    normalize_columns: bool = False,
    verbose: bool = False,
) -> NSPointwiseFitResult:
    """Fit a strong-form pointwise momentum equation from velocity, pressure, and density.

    The default fitted model is

    ``dt(v) = c_m (v . grad) v + c_nu Laplacian(v) + c_p grad(p) / rho``

    where the pressure contribution is included with a positive sign in the
    feature library, so the incompressible Navier-Stokes convention would
    usually recover a negative coefficient for ``grad(p) / rho``.

    ``include_terms`` can restrict the fitted feature library. Valid term names
    are ``linear_velocity``, ``quadratic_velocity``, ``material``,
    ``laplacian``, and ``pressure_gradient_over_density``.
    """

    def emit_progress(message: str) -> None:
        if verbose:
            print(message, flush=True)

    spatial_axes = tuple(spatial_axes)
    spacings = tuple(float(value) for value in spacings)
    included_feature_names = _validate_included_feature_names(
        include_terms,
        _NS_POINTWISE_FEATURE_NAMES,
        _NS_POINTWISE_DEFAULT_FEATURE_NAMES,
    )

    if velocity_field.shape[-1] != 3:
        raise ValueError("Velocity must have shape (..., 3).")
    if pressure_field.shape != velocity_field.shape[:-1]:
        raise ValueError(
            "pressure_field must share the time/space layout of velocity_field; got "
            f"pressure shape {pressure_field.shape} and velocity shape {velocity_field.shape}."
        )

    density_array = _normalize_density_field(density_field, pressure_field.shape)
    spatial_shape = tuple(velocity_field.shape[axis] for axis in spatial_axes)
    point_sample_slices = _normalize_pointwise_sample_slices(
        sample_slices,
        sample_step,
        spatial_shape,
    )

    emit_progress(
        "Starting pointwise NS fit: "
        f"velocity shape={velocity_field.shape}, pressure shape={pressure_field.shape}, "
        f"density shape={density_array.shape}, "
        f"spatial_derivative_method={spatial_derivative_method}, "
        f"time_derivative_method={time_derivative_method}, "
        f"sample_slices={point_sample_slices}, "
        f"include_terms={included_feature_names}"
    )
    emit_progress("Computing dt(v) and selected strong-form momentum terms...")

    dt_velocity = vector_time_derivative(
        velocity_field,
        dt=dt,
        time_axis=time_axis,
        periodic=False,
        time_derivative_method=time_derivative_method,
    )

    raw_feature_by_name: dict[str, np.ndarray] = {}
    if "linear_velocity" in included_feature_names:
        raw_feature_by_name["linear_velocity"] = np.asarray(velocity_field, dtype=float)
    if "quadratic_velocity" in included_feature_names:
        raw_feature_by_name["quadratic_velocity"] = aligned_quadratic_velocity(
            velocity_field
        )
    if "material" in included_feature_names:
        raw_feature_by_name["material"] = vector_material_derivative(
            velocity_field,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
    if "laplacian" in included_feature_names:
        raw_feature_by_name["laplacian"] = vector_laplacian(
            velocity_field,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
    if "pressure_gradient_over_density" in included_feature_names:
        pressure_gradient = _scalar_gradient(
            pressure_field,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
        raw_feature_by_name["pressure_gradient_over_density"] = (
            pressure_gradient / density_array[..., np.newaxis]
        )

    time_mask = _prepare_time_slice_mask(
        velocity_field.shape[time_axis],
        use_interior_time_only=use_interior_time_only,
    )
    masked_target = np.compress(time_mask, dt_velocity, axis=time_axis)
    regression_features = []
    for feature_name in included_feature_names:
        masked_feature = np.compress(
            time_mask,
            raw_feature_by_name[feature_name],
            axis=time_axis,
        )
        masked_feature = _apply_pointwise_sample_slices(
            masked_feature,
            point_sample_slices,
            spatial_axes,
        )
        regression_features.append(masked_feature)
    masked_target = _apply_pointwise_sample_slices(
        masked_target,
        point_sample_slices,
        spatial_axes,
    )
    regression_features = tuple(regression_features)
    feature_array = np.stack(regression_features, axis=-1)

    emit_progress(
        "Starting sparse pointwise regression: "
        f"target shape={masked_target.shape}, feature shape={feature_array.shape}"
    )
    coefficients, prediction, r2, rmse, relative_residual = _fit_linear_term_model(
        masked_target,
        regression_features,
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
    )
    emit_progress(
        "Finished pointwise NS fit: "
        f"r2={r2:.4f}, relative_residual={relative_residual:.4f}, "
        f"sample_count={int(masked_target.size)}"
    )

    coefficient_by_name = dict(zip(included_feature_names, coefficients))
    return NSPointwiseFitResult(
        dt_coefficient=1.0,
        material_coefficient=float(coefficient_by_name.get("material", 0.0)),
        viscosity_coefficient=float(coefficient_by_name.get("laplacian", 0.0)),
        pressure_gradient_coefficient=float(
            coefficient_by_name.get("pressure_gradient_over_density", 0.0)
        ),
        linear_velocity_coefficient=float(
            coefficient_by_name.get("linear_velocity", 0.0)
        ),
        quadratic_velocity_coefficient=float(
            coefficient_by_name.get("quadratic_velocity", 0.0)
        ),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(masked_target.size),
        pointwise_target=masked_target,
        pointwise_prediction=prediction.reshape(masked_target.shape),
        pointwise_features=feature_array,
        coefficients=coefficients,
        feature_names=included_feature_names,
        sample_slices=point_sample_slices,
    )


def _sampled_q_molecular_field(
    q_tensor_field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int],
    periodic: bool,
) -> np.ndarray:
    q_center = _take_spatial_points(q_tensor_field, sample_indices, spatial_axes)
    trace_q2 = np.einsum("...ij,...ji->...", q_center, q_center)
    q_squared = np.einsum("...ik,...kj->...ij", q_center, q_center)
    identity = np.eye(3, dtype=q_center.dtype)
    bulk_term = (
        -a2 * q_center
        - a3 * (q_squared - trace_q2[..., np.newaxis, np.newaxis] * identity / 3.0)
        - a4 * trace_q2[..., np.newaxis, np.newaxis] * q_center
    )
    return bulk_term + kappa * _sampled_tensor_laplacian(
        q_tensor_field,
        sample_indices,
        spacings,
        spatial_axes,
        periodic,
    )


def _sampled_passive_backflow_stress(
    q_tensor_field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
    flow_alignment_xi: float,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int],
    periodic: bool,
    include_distortion_stress: bool,
) -> np.ndarray:
    q_center = _take_spatial_points(q_tensor_field, sample_indices, spatial_axes)
    molecular_field = _sampled_q_molecular_field(
        q_tensor_field,
        sample_indices,
        a2=a2,
        a3=a3,
        a4=a4,
        kappa=kappa,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
    )
    trace_q2 = np.einsum("...ij,...ji->...", q_center, q_center)
    q_squared = np.einsum("...ik,...kj->...ij", q_center, q_center)
    trace_q3 = np.einsum("...ij,...ji->...", q_squared, q_center)
    q_grad_sq = np.zeros(q_center.shape[:-2], dtype=float)
    sampled_first_derivatives = []
    for derivative_axis_index, spacing in enumerate(spacings):
        q_partial = _sampled_first_derivative(
            q_tensor_field,
            sample_indices,
            spatial_axes,
            derivative_axis_index,
            spacing,
            periodic,
        )
        sampled_first_derivatives.append(q_partial)
        q_grad_sq += np.einsum("...ab,...ab->...", q_partial, q_partial)
    free_energy_density = (
        0.5 * a2 * trace_q2
        + (a3 / 3.0) * trace_q3
        + 0.25 * a4 * trace_q2**2
        + 0.5 * kappa * q_grad_sq
    )

    identity = np.eye(3, dtype=q_center.dtype)
    shifted_q = q_center + identity / 3.0
    q_contract_h = np.einsum("...ij,...ij->...", q_center, molecular_field)
    hq_t = np.einsum("...ac,...bc->...ab", molecular_field, q_center)
    qh_t = np.einsum("...ac,...bc->...ab", q_center, molecular_field)
    h_shifted_q_t = np.einsum("...ac,...bc->...ab", molecular_field, shifted_q)
    shifted_q_h_t = np.einsum("...ac,...bc->...ab", shifted_q, molecular_field)

    stress = -free_energy_density[..., np.newaxis, np.newaxis] * identity
    stress += (
        -2.0 * flow_alignment_xi * shifted_q * q_contract_h[..., np.newaxis, np.newaxis]
        + flow_alignment_xi * (h_shifted_q_t + shifted_q_h_t)
        + (hq_t - qh_t)
    )

    if include_distortion_stress:
        q_grad = np.stack(sampled_first_derivatives, axis=-3)
        distortion_stress = kappa * np.einsum(
            "...iab,...jab->...ij",
            q_grad,
            q_grad,
        )
        stress = stress + distortion_stress

    return stress


def _sampled_passive_backflow_force(
    q_tensor_field: np.ndarray,
    sample_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
    flow_alignment_xi: float,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int],
    periodic: bool,
    include_distortion_stress: bool,
) -> np.ndarray:
    center_stress = _sampled_passive_backflow_stress(
        q_tensor_field,
        sample_indices,
        a2=a2,
        a3=a3,
        a4=a4,
        kappa=kappa,
        flow_alignment_xi=flow_alignment_xi,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        include_distortion_stress=include_distortion_stress,
    )
    force = np.zeros(center_stress.shape[:-1], dtype=float)

    for derivative_axis_index, spacing in enumerate(spacings):
        axis_length = q_tensor_field.shape[spatial_axes[derivative_axis_index]]
        center_indices = sample_indices[derivative_axis_index]
        left_indices = _neighbor_indices(center_indices, axis_length, -1, periodic)
        right_indices = _neighbor_indices(center_indices, axis_length, 1, periodic)

        left_sample = list(sample_indices)
        right_sample = list(sample_indices)
        left_sample[derivative_axis_index] = left_indices
        right_sample[derivative_axis_index] = right_indices

        left_stress = _sampled_passive_backflow_stress(
            q_tensor_field,
            tuple(left_sample),
            a2=a2,
            a3=a3,
            a4=a4,
            kappa=kappa,
            flow_alignment_xi=flow_alignment_xi,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            include_distortion_stress=include_distortion_stress,
        )
        right_stress = _sampled_passive_backflow_stress(
            q_tensor_field,
            tuple(right_sample),
            a2=a2,
            a3=a3,
            a4=a4,
            kappa=kappa,
            flow_alignment_xi=flow_alignment_xi,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            include_distortion_stress=include_distortion_stress,
        )
        derivative = (
            right_stress[..., :, derivative_axis_index]
            - left_stress[..., :, derivative_axis_index]
        ) / (2.0 * spacing)

        if not periodic:
            start_mask = center_indices == 0
            end_mask = center_indices == axis_length - 1
            start_derivative = (
                right_stress[..., :, derivative_axis_index]
                - center_stress[..., :, derivative_axis_index]
            ) / spacing
            end_derivative = (
                center_stress[..., :, derivative_axis_index]
                - left_stress[..., :, derivative_axis_index]
            ) / spacing
            output_axis = spatial_axes[derivative_axis_index] % derivative.ndim
            _replace_axis_values(derivative, start_derivative, output_axis, start_mask)
            _replace_axis_values(derivative, end_derivative, output_axis, end_mask)

        force += derivative

    return -force


def fit_be_equation_velocity_pointwise(
    velocity_field: np.ndarray,
    pressure_field: np.ndarray,
    density_field: np.ndarray | float,
    q_tensor_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
    flow_alignment_xi: float = 1.0,
    include_distortion_stress: bool = True,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2, 3),
    periodic: bool = False,
    spatial_derivative_method: str = "finite_difference",
    time_derivative_method: str = "central",
    use_interior_time_only: bool = True,
    include_terms: Sequence[str] | None = None,
    sample_step: int | Sequence[int] | None = None,
    sample_slices: Sequence[object] | None = None,
    threshold: float = 1e-10,
    alpha: float = 0.0,
    normalize_columns: bool = False,
    verbose: bool = False,
) -> BEPointwiseVelocityFitResult:
    """Fit a strong-form pointwise BE-style momentum equation from Q and velocity.

    The default fitted model is

    ``dt(v) = c_m (v . grad) v + c_nu Laplacian(v) + c_p grad(p)/rho
              + c_a div(Q) + c_b [-div(sigma_passive(Q, H))]``

    where ``H`` and ``sigma_passive`` follow the solver conventions supplied by
    the caller through ``a2``, ``a3``, ``a4``, ``kappa``, and
    ``flow_alignment_xi``.
    """

    def emit_progress(message: str) -> None:
        if verbose:
            print(message, flush=True)

    spatial_axes = tuple(spatial_axes)
    spacings = tuple(float(value) for value in spacings)
    included_feature_names = _validate_included_feature_names(
        include_terms,
        _BE_POINTWISE_FEATURE_NAMES,
        _BE_POINTWISE_DEFAULT_FEATURE_NAMES,
    )

    if velocity_field.shape[-1] != 3:
        raise ValueError("Velocity must have shape (..., 3).")
    if q_tensor_field.shape[:-2] != velocity_field.shape[:-1]:
        raise ValueError(
            "Q must share the time/space layout of the velocity field and end with "
            f"(3, 3); got velocity shape {velocity_field.shape} and Q shape "
            f"{q_tensor_field.shape}."
        )
    if q_tensor_field.shape[-2:] != (3, 3):
        raise ValueError("The last two axes of Q must each have length 3.")
    if pressure_field.shape != velocity_field.shape[:-1]:
        raise ValueError(
            "pressure_field must share the time/space layout of velocity_field; got "
            f"pressure shape {pressure_field.shape} and velocity shape {velocity_field.shape}."
        )

    density_array = _normalize_density_field(density_field, pressure_field.shape)
    spatial_shape = tuple(velocity_field.shape[axis] for axis in spatial_axes)
    point_sample_slices = _normalize_pointwise_sample_slices(
        sample_slices,
        sample_step,
        spatial_shape,
    )
    sample_indices = None
    if point_sample_slices is not None:
        sample_indices = _pointwise_sample_indices(point_sample_slices, spatial_shape)
    if (
        point_sample_slices is not None
        and spatial_derivative_method != "finite_difference"
    ):
        raise ValueError(
            "sample_step/sample_slices currently support only "
            "spatial_derivative_method='finite_difference'."
        )

    emit_progress(
        "Starting pointwise BE velocity fit: "
        f"velocity shape={velocity_field.shape}, pressure shape={pressure_field.shape}, "
        f"density shape={density_array.shape}, Q shape={q_tensor_field.shape}, "
        f"spatial_derivative_method={spatial_derivative_method}, "
        f"time_derivative_method={time_derivative_method}, "
        f"sample_slices={point_sample_slices}, "
        f"include_terms={included_feature_names}"
    )
    emit_progress(
        "Computing dt(v), selected hydrodynamic terms, div(Q), and passive backflow..."
    )

    dt_velocity = vector_time_derivative(
        velocity_field,
        dt=dt,
        time_axis=time_axis,
        periodic=False,
        time_derivative_method=time_derivative_method,
    )

    raw_feature_by_name: dict[str, np.ndarray] = {}
    pre_sampled_feature_names: set[str] = set()
    if "linear_velocity" in included_feature_names:
        raw_feature_by_name["linear_velocity"] = np.asarray(velocity_field, dtype=float)
    if "quadratic_velocity" in included_feature_names:
        raw_feature_by_name["quadratic_velocity"] = aligned_quadratic_velocity(
            velocity_field
        )
    if sample_indices is None:
        if "material" in included_feature_names:
            raw_feature_by_name["material"] = vector_material_derivative(
                velocity_field,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=spatial_derivative_method,
            )
        if "laplacian" in included_feature_names:
            raw_feature_by_name["laplacian"] = vector_laplacian(
                velocity_field,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=spatial_derivative_method,
            )
        if "pressure_gradient_over_density" in included_feature_names:
            pressure_gradient = _scalar_gradient(
                pressure_field,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=spatial_derivative_method,
            )
            raw_feature_by_name["pressure_gradient_over_density"] = (
                pressure_gradient / density_array[..., np.newaxis]
            )
        if "q_divergence" in included_feature_names:
            raw_feature_by_name["q_divergence"] = active_q_divergence_force(
                q_tensor_field,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=spatial_derivative_method,
            )
    else:
        if "material" in included_feature_names:
            raw_feature_by_name["material"] = _sampled_vector_material_derivative(
                velocity_field,
                sample_indices,
                spacings,
                spatial_axes,
                periodic,
            )
            pre_sampled_feature_names.add("material")
        if "laplacian" in included_feature_names:
            raw_feature_by_name["laplacian"] = _sampled_vector_laplacian(
                velocity_field,
                sample_indices,
                spacings,
                spatial_axes,
                periodic,
            )
            pre_sampled_feature_names.add("laplacian")
        if "pressure_gradient_over_density" in included_feature_names:
            pressure_gradient = _sampled_scalar_gradient(
                pressure_field,
                sample_indices,
                spacings,
                spatial_axes,
                periodic,
            )
            sampled_density = _take_spatial_points(
                density_array,
                sample_indices,
                spatial_axes,
            )
            raw_feature_by_name["pressure_gradient_over_density"] = (
                pressure_gradient / sampled_density[..., np.newaxis]
            )
            pre_sampled_feature_names.add("pressure_gradient_over_density")
        if "q_divergence" in included_feature_names:
            raw_feature_by_name["q_divergence"] = _sampled_tensor_divergence(
                q_tensor_field,
                sample_indices,
                spacings,
                spatial_axes,
                periodic,
            )
            pre_sampled_feature_names.add("q_divergence")
    if "passive_backflow" in included_feature_names:
        if sample_indices is None:
            raw_feature_by_name["passive_backflow"] = passive_backflow_force(
                q_tensor_field,
                a2=a2,
                a3=a3,
                a4=a4,
                kappa=kappa,
                flow_alignment_xi=flow_alignment_xi,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=spatial_derivative_method,
                include_distortion_stress=include_distortion_stress,
            )
        else:
            canonical_q = np.moveaxis(
                q_tensor_field,
                (time_axis, *spatial_axes, q_tensor_field.ndim - 2, q_tensor_field.ndim - 1),
                (0, 1, 2, 3, 4, 5),
            )
            sampled_force_slices = []
            for q_slice in canonical_q:
                sampled_force_slices.append(
                    _sampled_passive_backflow_force(
                        q_slice,
                        sample_indices,
                        a2=a2,
                        a3=a3,
                        a4=a4,
                        kappa=kappa,
                        flow_alignment_xi=flow_alignment_xi,
                        spacings=spacings,
                        spatial_axes=(0, 1, 2),
                        periodic=periodic,
                        include_distortion_stress=include_distortion_stress,
                    )
                )
            raw_feature_by_name["passive_backflow"] = np.stack(
                sampled_force_slices,
                axis=0,
            )
            pre_sampled_feature_names.add("passive_backflow")

    time_mask = _prepare_time_slice_mask(
        velocity_field.shape[time_axis],
        use_interior_time_only=use_interior_time_only,
    )
    masked_target = np.compress(time_mask, dt_velocity, axis=time_axis)
    regression_features = []
    for feature_name in included_feature_names:
        masked_feature = np.compress(
            time_mask,
            raw_feature_by_name[feature_name],
            axis=time_axis,
        )
        if sample_indices is None or feature_name in pre_sampled_feature_names:
            masked_feature = _apply_pointwise_sample_slices(
                masked_feature,
                None if feature_name in pre_sampled_feature_names else point_sample_slices,
                spatial_axes,
            )
        else:
            masked_feature = _apply_pointwise_sample_slices(
                masked_feature,
                point_sample_slices,
                spatial_axes,
            )
        regression_features.append(masked_feature)
    masked_target = _apply_pointwise_sample_slices(
        masked_target,
        point_sample_slices,
        spatial_axes,
    )
    regression_features = tuple(regression_features)
    feature_array = np.stack(regression_features, axis=-1)

    emit_progress(
        "Starting sparse pointwise regression: "
        f"target shape={masked_target.shape}, feature shape={feature_array.shape}"
    )
    coefficients, prediction, r2, rmse, relative_residual = _fit_linear_term_model(
        masked_target,
        regression_features,
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
    )
    emit_progress(
        "Finished pointwise BE velocity fit: "
        f"r2={r2:.4f}, relative_residual={relative_residual:.4f}, "
        f"sample_count={int(masked_target.size)}"
    )

    coefficient_by_name = dict(zip(included_feature_names, coefficients))
    return BEPointwiseVelocityFitResult(
        dt_coefficient=1.0,
        material_coefficient=float(coefficient_by_name.get("material", 0.0)),
        viscosity_coefficient=float(coefficient_by_name.get("laplacian", 0.0)),
        pressure_gradient_coefficient=float(
            coefficient_by_name.get("pressure_gradient_over_density", 0.0)
        ),
        q_divergence_coefficient=float(
            coefficient_by_name.get("q_divergence", 0.0)
        ),
        passive_backflow_coefficient=float(
            coefficient_by_name.get("passive_backflow", 0.0)
        ),
        linear_velocity_coefficient=float(
            coefficient_by_name.get("linear_velocity", 0.0)
        ),
        quadratic_velocity_coefficient=float(
            coefficient_by_name.get("quadratic_velocity", 0.0)
        ),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(masked_target.size),
        pointwise_target=masked_target,
        pointwise_prediction=prediction.reshape(masked_target.shape),
        pointwise_features=feature_array,
        coefficients=coefficients,
        feature_names=included_feature_names,
        sample_slices=point_sample_slices,
    )


def fit_be_equation_velocity(
    velocity_field: np.ndarray,
    q_tensor_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    spatial_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    test_fields: Sequence[np.ndarray] | None = None,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2, 3),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    time_derivative_method: str = "central",
    use_interior_time_only: bool = True,
    test_field_specs: Sequence[TestFieldSpec] = DEFAULT_TEST_FIELD_SPECS,
    threshold: float = 1e-10,
    alpha: float = 0.0,
    normalize_columns: bool = False,
) -> BEEquationVelocityFitResult:
    """Fit BE-style velocity dynamics with a weak form and an added div(Q) term.

    This is a global weak-form checker intended mainly for periodic settings
    with globally defined divergence-free test fields. The pressure term drops
    out when paired with divergence-free test functions.

    ``spatial_derivative_method`` controls how the spatial derivatives in
    ``(v · grad) v``, ``Laplacian(v)``, and ``div(Q)`` are computed. The
    default is centered finite differences. Setting
    ``spatial_derivative_method="spectral"`` switches to periodic Fourier
    derivatives and therefore requires ``periodic=True``.
    """
    spatial_axes = tuple(spatial_axes)
    spacings = tuple(float(value) for value in spacings)
    test_field_specs = tuple(test_field_specs)

    if q_tensor_field.shape[:-2] != velocity_field.shape[:-1]:
        raise ValueError(
            "Q must share the time/space layout of the velocity field and end with "
            f"(3, 3); got velocity shape {velocity_field.shape} and Q shape "
            f"{q_tensor_field.shape}."
        )
    if q_tensor_field.shape[-2:] != (3, 3):
        raise ValueError("The last two axes of Q must each have length 3.")

    if spatial_grid is None:
        spatial_grid = _spatial_grid_from_field(velocity_field, spacings, spatial_axes)

    dt_velocity = vector_time_derivative(
        velocity_field,
        dt=dt,
        time_axis=time_axis,
        periodic=False,
        time_derivative_method=time_derivative_method,
    )
    linear_velocity = np.asarray(velocity_field, dtype=float)
    quadratic_velocity = aligned_quadratic_velocity(velocity_field)
    material_velocity = vector_material_derivative(
        velocity_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    laplacian_velocity = vector_laplacian(
        velocity_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    q_divergence = tensor_divergence(
        q_tensor_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )

    weak_targets: list[np.ndarray] = []
    weak_linear_velocity: list[np.ndarray] = []
    weak_quadratic_velocity: list[np.ndarray] = []
    weak_material: list[np.ndarray] = []
    weak_laplacian: list[np.ndarray] = []
    weak_q_divergence: list[np.ndarray] = []

    if test_fields is None:
        test_field_bank = [
            build_beltrami_test_field(*spatial_grid, spec) for spec in test_field_specs
        ]
    else:
        test_field_bank = [np.asarray(field, dtype=float) for field in test_fields]
        expected_shape = tuple(velocity_field.shape[axis] for axis in spatial_axes) + (3,)
        for field in test_field_bank:
            if field.shape != expected_shape:
                raise ValueError(
                    "Each custom test field must have shape "
                    f"{expected_shape}, got {field.shape}."
                )

    for test_field_spatial in test_field_bank:
        test_field = np.broadcast_to(
            test_field_spatial[np.newaxis, ...],
            velocity_field.shape,
        )
        weak_targets.append(
            weak_inner_product(
                test_field,
                dt_velocity,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
        )
        weak_linear_velocity.append(
            weak_inner_product(
                test_field,
                linear_velocity,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
        )
        weak_quadratic_velocity.append(
            weak_inner_product(
                test_field,
                quadratic_velocity,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
        )
        weak_material.append(
            weak_inner_product(
                test_field,
                material_velocity,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
        )
        weak_laplacian.append(
            weak_inner_product(
                test_field,
                laplacian_velocity,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
        )
        weak_q_divergence.append(
            weak_inner_product(
                test_field,
                q_divergence,
                spacings=spacings,
                spatial_axes=spatial_axes,
            )
        )

    weak_target = np.stack(weak_targets, axis=0)
    weak_linear_velocity_feature = np.stack(weak_linear_velocity, axis=0)
    weak_quadratic_velocity_feature = np.stack(weak_quadratic_velocity, axis=0)
    weak_material_feature = np.stack(weak_material, axis=0)
    weak_laplacian_feature = np.stack(weak_laplacian, axis=0)
    weak_q_divergence_feature = np.stack(weak_q_divergence, axis=0)

    time_mask = _prepare_time_slice_mask(
        weak_target.shape[1],
        use_interior_time_only=use_interior_time_only,
    )
    masked_target = weak_target[:, time_mask]
    masked_linear_velocity = weak_linear_velocity_feature[:, time_mask]
    masked_quadratic_velocity = weak_quadratic_velocity_feature[:, time_mask]
    masked_material = weak_material_feature[:, time_mask]
    masked_laplacian = weak_laplacian_feature[:, time_mask]
    masked_q_divergence = weak_q_divergence_feature[:, time_mask]

    coefficients, prediction, r2, rmse, relative_residual = _fit_linear_term_model(
        masked_target,
        (
            masked_linear_velocity,
            masked_quadratic_velocity,
            masked_material,
            masked_laplacian,
            masked_q_divergence,
        ),
        threshold=threshold,
        alpha=alpha,
        normalize_columns=normalize_columns,
    )

    return BEEquationVelocityFitResult(
        material_coefficient=float(coefficients[2]),
        viscosity_coefficient=float(coefficients[3]),
        q_divergence_coefficient=float(coefficients[4]),
        linear_velocity_coefficient=float(coefficients[0]),
        quadratic_velocity_coefficient=float(coefficients[1]),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(masked_target.size),
        weak_target=masked_target,
        weak_prediction=prediction.reshape(masked_target.shape),
        weak_features=np.stack(
            [
                masked_linear_velocity,
                masked_quadratic_velocity,
                masked_material,
                masked_laplacian,
                masked_q_divergence,
            ],
            axis=-1,
        ),
        coefficients=coefficients,
        feature_names=(
            "linear_velocity",
            "quadratic_velocity",
            "material",
            "laplacian",
            "q_divergence",
        ),
        test_field_specs=test_field_specs if test_fields is None else (),
    )


def fit_be_equation_velocity_local_weak_form(
    velocity_field: np.ndarray,
    q_tensor_field: np.ndarray,
    dt: float,
    spacings: Sequence[float],
    *,
    viscosity: float = 1.0,
    a2: float | None = None,
    a3: float | None = None,
    a4: float | None = None,
    kappa: float | None = None,
    flow_alignment_xi: float = 1.0,
    include_distortion_stress: bool = True,
    include_passive_backflow: bool = False,
    patch_half_widths: tuple[int, int, int] = (3, 3, 3),
    patch_strides: tuple[int, int, int] = (4, 4, 4),
    patch_specs: Sequence[LocalPatchSpec] | None = None,
    time_axis: int = 0,
    spatial_axes: Sequence[int] = (1, 2, 3),
    periodic: bool = False,
    spatial_derivative_method: str = "finite_difference",
    time_derivative_method: str = "central",
    num_test_fields_per_patch: int = 1,
    use_interior_time_only: bool = True,
    threshold: float = 1e-10,
    optimizer_alpha: float = 0.0,
    normalize_columns: bool = False,
    verbose: bool = False,
    progress_interval: int = 10,
) -> BELocalEquationVelocityFitResult:
    """Fit local weak-form coefficients for a velocity equation with ``div(Q)`` forcing.

    The fitted model is, by default,

    ``Laplacian(v) = c_dt * dt(v) + c_material * (v · grad) v + alpha * div(Q)``

    where ``v`` is a 3D velocity field and ``Q`` is a known 3x3 tensor field.
    The regression samples are built from compactly supported divergence-free
    test fields placed on many local spatial patches, so the pressure term is
    removed in weak form rather than modeled explicitly.

    Parameters
    ----------
    velocity_field:
        Velocity data with shape ``(nt, nx, ny, nz, 3)`` under the default axis
        convention. The last axis must contain the three vector components.
    q_tensor_field:
        Tensor data aligned with ``velocity_field`` and shaped
        ``(nt, nx, ny, nz, 3, 3)`` under the default axis convention. The last
        two axes store the tensor row and column indices.
    dt:
        Time spacing between successive stored frames in ``velocity_field``.
    spacings:
        Spatial grid spacings ``(dx, dy, dz)`` corresponding to
        ``spatial_axes``.
    viscosity:
        Physical kinematic viscosity used to rescale the viscosity-normalized
        regression coefficients for reporting. The default is ``1.0``.
    patch_half_widths:
        Half-width of each local patch, measured in grid points along each
        spatial axis. Larger values produce wider compact-support test fields
        and therefore more spatial averaging.
    patch_strides:
        Stride between neighboring patch centers, measured in grid points along
        each spatial axis. Smaller values create more patches and more
        regression samples.
    patch_specs:
        Optional explicit patch specification sequence. If provided, it
        overrides ``patch_half_widths`` and ``patch_strides``.
    time_axis:
        Axis index of time in ``velocity_field`` and ``q_tensor_field``.
    spatial_axes:
        Three axis indices corresponding to the spatial coordinates.
    periodic:
        Whether the spatial derivative operators should assume periodic
        boundary conditions. The local weak-form construction itself does not
        require periodicity, but the derivative backend will use this flag.
    spatial_derivative_method:
        Spatial derivative backend used for ``(v · grad) v``, ``Laplacian(v)``,
        and ``div(Q)``. The default ``"finite_difference"`` uses centered
        finite differences. ``"spectral"`` uses periodic Fourier derivatives
        and therefore requires ``periodic=True``.
    time_derivative_method:
        Time-derivative backend used for ``dt(v)``. ``"central"`` is the
        default continuous-PDE-style estimate. ``"forward"`` is often a better
        match to a step-by-step time-marching solver.
    num_test_fields_per_patch:
        Number of deterministic divergence-free test fields generated per patch.
        Supported values are ``1``, ``3``, ``12``, and ``30``.
    use_interior_time_only:
        If ``True``, discard the first and last time slices from the regression
        because the centered finite-difference time derivative is least
        symmetric at the temporal boundaries.
    threshold:
        Sparsity threshold passed to the PySINDy ``STLSQ`` optimizer.
    optimizer_alpha:
        Ridge-style regularization strength passed to the PySINDy ``STLSQ``
        optimizer. This is an optimizer hyperparameter, not the physical
        coefficient multiplying ``div(Q)`` in the fitted equation.
    normalize_columns:
        Whether to normalize feature columns before sparse regression.
    verbose:
        If ``True``, print human-readable progress updates while building the
        local weak-form samples and fitting the regression model.
    progress_interval:
        Number of completed patches between progress updates when
        ``verbose=True``. Smaller values print more frequently.

    Returns
    -------
    BELocalEquationVelocityFitResult
        Dataclass containing the recovered coefficients, regression quality
        metrics, patch-wise weak-form targets and features, and the patch
        definitions actually used. The returned ``dt_coefficient`` and
        ``viscosity_coefficient`` reflect the chosen normalization mode.
    """
    def emit_progress(message: str) -> None:
        if verbose:
            print(message, flush=True)

    overall_start = time.perf_counter()
    spatial_axes = tuple(spatial_axes)
    spacings = tuple(float(value) for value in spacings)
    viscosity = float(viscosity)
    if not np.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("viscosity must be a positive finite value.")
    normalization_mode = "viscosity"
    num_test_fields_per_patch = _validate_num_test_fields_per_patch(
        num_test_fields_per_patch
    )
    if include_passive_backflow:
        missing_parameters = [
            name
            for name, value in (
                ("a2", a2),
                ("a3", a3),
                ("a4", a4),
                ("kappa", kappa),
            )
            if value is None
        ]
        if missing_parameters:
            raise ValueError(
                "include_passive_backflow=True requires parameters "
                + ", ".join(missing_parameters)
                + "."
            )

    if q_tensor_field.shape[:-2] != velocity_field.shape[:-1]:
        raise ValueError(
            "Q must share the time/space layout of the velocity field and end with "
            f"(3, 3); got velocity shape {velocity_field.shape} and Q shape "
            f"{q_tensor_field.shape}."
        )
    if q_tensor_field.shape[-2:] != (3, 3):
        raise ValueError("The last two axes of Q must each have length 3.")

    if spatial_derivative_method.lower() == "spectral":
        raise ValueError(
            "fit_be_equation_velocity_local_weak_form evaluates features patch by patch; "
            "the spectral derivative backend is not supported in this local mode."
        )

    emit_progress(
        "Starting local weak-form fit: "
        f"velocity shape={velocity_field.shape}, Q shape={q_tensor_field.shape}, "
        f"normalization_mode={normalization_mode}, "
        f"spatial_derivative_method={spatial_derivative_method}, "
        f"time_derivative_method={time_derivative_method}, "
        f"num_test_fields_per_patch={num_test_fields_per_patch}"
    )
    backflow_message = ", and passive backflow" if include_passive_backflow else ""
    emit_progress(
        "Computing dt(v), v, |v|v, (v·grad)v, Laplacian(v), and div(Q)"
        f"{backflow_message} patch by patch..."
    )

    spatial_shape = tuple(velocity_field.shape[axis] for axis in spatial_axes)
    if patch_specs is None:
        patch_specs = generate_local_patch_specs(
            shape=spatial_shape,
            half_widths=patch_half_widths,
            strides=patch_strides,
        )
    else:
        patch_specs = tuple(patch_specs)

    patch_count = len(patch_specs)
    halo_widths = (2, 2, 2) if include_passive_backflow else (1, 1, 1)
    emit_progress(
        "Starting local patch integration: "
        f"{patch_count} patches, half_widths={patch_half_widths}, strides={patch_strides}"
    )

    local_targets: list[np.ndarray] = []
    local_linear_velocity: list[np.ndarray] = []
    local_quadratic_velocity: list[np.ndarray] = []
    local_material: list[np.ndarray] = []
    local_laplacian: list[np.ndarray] = []
    local_q_divergence: list[np.ndarray] = []
    local_passive_backflow: list[np.ndarray] = []

    patch_loop_start = time.perf_counter()
    for patch_index, patch in enumerate(patch_specs, start=1):
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
            local_velocity_field.shape[axis] for axis in spatial_axes
        )

        patch_dt_velocity = vector_time_derivative(
            local_velocity_field,
            dt=dt,
            time_axis=time_axis,
            periodic=False,
            time_derivative_method=time_derivative_method,
        )
        patch_linear_velocity = np.asarray(local_velocity_field, dtype=float)
        patch_quadratic_velocity = aligned_quadratic_velocity(local_velocity_field)
        patch_material_velocity = vector_material_derivative(
            local_velocity_field,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
        patch_laplacian_velocity = vector_laplacian(
            local_velocity_field,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
        patch_q_divergence = tensor_divergence(
            local_q_tensor_field,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
        patch_passive_backflow = None
        if include_passive_backflow:
            patch_passive_backflow = passive_backflow_force(
                local_q_tensor_field,
                a2=float(a2),
                a3=float(a3),
                a4=float(a4),
                kappa=float(kappa),
                flow_alignment_xi=flow_alignment_xi,
                spacings=spacings,
                spatial_axes=spatial_axes,
                periodic=periodic,
                spatial_derivative_method=spatial_derivative_method,
                include_distortion_stress=include_distortion_stress,
            )

        test_fields = build_local_divergence_free_test_fields(
            shape=local_spatial_shape,
            center_indices=local_patch.center_indices,
            half_widths=local_patch.half_widths,
            spacings=spacings,
            num_test_fields_per_patch=num_test_fields_per_patch,
        )
        for test_field_spatial in test_fields:
            test_field = np.broadcast_to(
                test_field_spatial[np.newaxis, ...],
                local_velocity_field.shape,
            )
            local_targets.append(
                weak_inner_product(
                    test_field,
                    patch_dt_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_linear_velocity.append(
                weak_inner_product(
                    test_field,
                    patch_linear_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_quadratic_velocity.append(
                weak_inner_product(
                    test_field,
                    patch_quadratic_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_material.append(
                weak_inner_product(
                    test_field,
                    patch_material_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_laplacian.append(
                weak_inner_product(
                    test_field,
                    patch_laplacian_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_q_divergence.append(
                weak_inner_product(
                    test_field,
                    patch_q_divergence,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            if patch_passive_backflow is not None:
                local_passive_backflow.append(
                    weak_inner_product(
                        test_field,
                        patch_passive_backflow,
                        spacings=spacings,
                        spatial_axes=spatial_axes,
                    )
                )

        should_report = (
            patch_index == 1
            or patch_index == patch_count
            or patch_index % max(progress_interval, 1) == 0
        )
        if should_report:
            elapsed = time.perf_counter() - patch_loop_start
            average_per_patch = elapsed / patch_index
            remaining = max(patch_count - patch_index, 0) * average_per_patch
            emit_progress(
                "Completed patch "
                f"{patch_index}/{patch_count} "
                f"(elapsed={elapsed:.1f}s, eta={remaining:.1f}s)"
            )

    emit_progress("Building regression arrays from local weak-form samples...")
    local_target = np.stack(local_targets, axis=0)
    local_linear_velocity_feature = np.stack(local_linear_velocity, axis=0)
    local_quadratic_velocity_feature = np.stack(local_quadratic_velocity, axis=0)
    local_material_feature = np.stack(local_material, axis=0)
    local_laplacian_feature = np.stack(local_laplacian, axis=0)
    local_q_divergence_feature = np.stack(local_q_divergence, axis=0)
    local_passive_backflow_feature = (
        np.stack(local_passive_backflow, axis=0)
        if include_passive_backflow
        else None
    )

    time_mask = _prepare_time_slice_mask(
        local_target.shape[1],
        use_interior_time_only=use_interior_time_only,
    )
    masked_target = local_target[:, time_mask]
    masked_linear_velocity = local_linear_velocity_feature[:, time_mask]
    masked_quadratic_velocity = local_quadratic_velocity_feature[:, time_mask]
    masked_material = local_material_feature[:, time_mask]
    masked_laplacian = local_laplacian_feature[:, time_mask]
    masked_q_divergence = local_q_divergence_feature[:, time_mask]
    masked_passive_backflow = (
        local_passive_backflow_feature[:, time_mask]
        if local_passive_backflow_feature is not None
        else None
    )

    if normalization_mode == "dt":
        regression_target = masked_target
        regression_features_list = [
            masked_linear_velocity,
            masked_quadratic_velocity,
            masked_material,
            masked_laplacian,
            masked_q_divergence,
        ]
        if masked_passive_backflow is not None:
            regression_features_list.append(masked_passive_backflow)
        regression_features = tuple(regression_features_list)
    else:
        regression_target = masked_laplacian
        regression_features_list = [
            masked_target,
            masked_linear_velocity,
            masked_quadratic_velocity,
            masked_material,
            masked_q_divergence,
        ]
        if masked_passive_backflow is not None:
            regression_features_list.append(masked_passive_backflow)
        regression_features = tuple(regression_features_list)

    feature_array = np.stack(regression_features, axis=-1)
    emit_progress(
        "Starting sparse regression: "
        f"target shape={regression_target.shape}, feature shape={feature_array.shape}"
    )
    coefficients, _prediction, r2, rmse, relative_residual = _fit_linear_term_model(
        regression_target,
        regression_features,
        threshold=threshold,
        alpha=optimizer_alpha,
        normalize_columns=normalize_columns,
    )
    emit_progress(
        "Finished local weak-form fit: "
        f"elapsed={time.perf_counter() - overall_start:.1f}s, "
        f"r2={r2:.4f}, relative_residual={relative_residual:.4f}"
    )

    return BELocalEquationVelocityFitResult(
        dt_coefficient=float(viscosity * coefficients[0]),
        material_coefficient=float(-viscosity * coefficients[3]),
        viscosity_coefficient=viscosity,
        q_divergence_coefficient=float(-viscosity * coefficients[4]),
        passive_backflow_coefficient=float(
            -viscosity * coefficients[5] if include_passive_backflow else 0.0
        ),
        linear_velocity_coefficient=float(-viscosity * coefficients[1]),
        quadratic_velocity_coefficient=float(-viscosity * coefficients[2]),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(masked_target.size),
    )

    emit_progress(
        "Starting local weak-form fit: "
        f"velocity shape={velocity_field.shape}, Q shape={q_tensor_field.shape}, "
        f"normalization_mode={normalization_mode}, "
        f"spatial_derivative_method={spatial_derivative_method}, "
        f"time_derivative_method={time_derivative_method}, "
        f"num_test_fields_per_patch={num_test_fields_per_patch}"
    )
    emit_progress("Computing dt(v), v, |v|v, (v·grad)v, Laplacian(v), and div(Q)...")

    dt_velocity = vector_time_derivative(
        velocity_field,
        dt=dt,
        time_axis=time_axis,
        periodic=False,
        time_derivative_method=time_derivative_method,
    )
    linear_velocity = np.asarray(velocity_field, dtype=float)
    quadratic_velocity = aligned_quadratic_velocity(velocity_field)
    material_velocity = vector_material_derivative(
        velocity_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    laplacian_velocity = vector_laplacian(
        velocity_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    q_divergence = tensor_divergence(
        q_tensor_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )

    spatial_shape = tuple(velocity_field.shape[axis] for axis in spatial_axes)
    if patch_specs is None:
        patch_specs = generate_local_patch_specs(
            shape=spatial_shape,
            half_widths=patch_half_widths,
            strides=patch_strides,
        )
    else:
        patch_specs = tuple(patch_specs)

    patch_count = len(patch_specs)
    emit_progress(
        "Starting local patch integration: "
        f"{patch_count} patches, half_widths={patch_half_widths}, strides={patch_strides}"
    )

    local_targets: list[np.ndarray] = []
    local_linear_velocity: list[np.ndarray] = []
    local_quadratic_velocity: list[np.ndarray] = []
    local_material: list[np.ndarray] = []
    local_laplacian: list[np.ndarray] = []
    local_q_divergence: list[np.ndarray] = []

    patch_loop_start = time.perf_counter()
    for patch_index, patch in enumerate(patch_specs, start=1):
        test_fields = build_local_divergence_free_test_fields(
            shape=spatial_shape,
            center_indices=patch.center_indices,
            half_widths=patch.half_widths,
            spacings=spacings,
            num_test_fields_per_patch=num_test_fields_per_patch,
        )
        for test_field_spatial in test_fields:
            test_field = np.broadcast_to(
                test_field_spatial[np.newaxis, ...],
                velocity_field.shape,
            )
            local_targets.append(
                weak_inner_product(
                    test_field,
                    dt_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_linear_velocity.append(
                weak_inner_product(
                    test_field,
                    linear_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_quadratic_velocity.append(
                weak_inner_product(
                    test_field,
                    quadratic_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_material.append(
                weak_inner_product(
                    test_field,
                    material_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_laplacian.append(
                weak_inner_product(
                    test_field,
                    laplacian_velocity,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )
            local_q_divergence.append(
                weak_inner_product(
                    test_field,
                    q_divergence,
                    spacings=spacings,
                    spatial_axes=spatial_axes,
                )
            )

        should_report = (
            patch_index == 1
            or patch_index == patch_count
            or patch_index % max(progress_interval, 1) == 0
        )
        if should_report:
            elapsed = time.perf_counter() - patch_loop_start
            average_per_patch = elapsed / patch_index
            remaining = max(patch_count - patch_index, 0) * average_per_patch
            emit_progress(
                "Completed patch "
                f"{patch_index}/{patch_count} "
                f"(elapsed={elapsed:.1f}s, eta={remaining:.1f}s)"
            )

    emit_progress("Building regression arrays from local weak-form samples...")
    local_target = np.stack(local_targets, axis=0)
    local_linear_velocity_feature = np.stack(local_linear_velocity, axis=0)
    local_quadratic_velocity_feature = np.stack(local_quadratic_velocity, axis=0)
    local_material_feature = np.stack(local_material, axis=0)
    local_laplacian_feature = np.stack(local_laplacian, axis=0)
    local_q_divergence_feature = np.stack(local_q_divergence, axis=0)

    time_mask = _prepare_time_slice_mask(
        local_target.shape[1],
        use_interior_time_only=use_interior_time_only,
    )
    masked_target = local_target[:, time_mask]
    masked_linear_velocity = local_linear_velocity_feature[:, time_mask]
    masked_quadratic_velocity = local_quadratic_velocity_feature[:, time_mask]
    masked_material = local_material_feature[:, time_mask]
    masked_laplacian = local_laplacian_feature[:, time_mask]
    masked_q_divergence = local_q_divergence_feature[:, time_mask]

    if normalization_mode == "dt":
        regression_target = masked_target
        regression_features = (
            masked_linear_velocity,
            masked_quadratic_velocity,
            masked_material,
            masked_laplacian,
            masked_q_divergence,
        )
    else:
        regression_target = masked_laplacian
        regression_features = (
            masked_target,
            masked_linear_velocity,
            masked_quadratic_velocity,
            masked_material,
            masked_q_divergence,
        )

    feature_array = np.stack(regression_features, axis=-1)
    emit_progress(
        "Starting sparse regression: "
        f"target shape={regression_target.shape}, feature shape={feature_array.shape}"
    )
    coefficients, _prediction, r2, rmse, relative_residual = _fit_linear_term_model(
        regression_target,
        regression_features,
        threshold=threshold,
        alpha=optimizer_alpha,
        normalize_columns=normalize_columns,
    )
    emit_progress(
        "Finished local weak-form fit: "
        f"elapsed={time.perf_counter() - overall_start:.1f}s, "
        f"r2={r2:.4f}, relative_residual={relative_residual:.4f}"
    )

    return BELocalEquationVelocityFitResult(
        dt_coefficient=float(viscosity * coefficients[0]),
        material_coefficient=float(-viscosity * coefficients[3]),
        viscosity_coefficient=viscosity,
        q_divergence_coefficient=float(-viscosity * coefficients[4]),
        passive_backflow_coefficient=0.0,
        linear_velocity_coefficient=float(-viscosity * coefficients[1]),
        quadratic_velocity_coefficient=float(-viscosity * coefficients[2]),
        r2=r2,
        rmse=rmse,
        relative_residual=relative_residual,
        sample_count=int(regression_target.size),
    )
