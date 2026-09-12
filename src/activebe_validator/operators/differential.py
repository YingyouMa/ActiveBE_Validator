"""Finite-difference and spectral differential operators on structured grids."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _validate_spatial_derivative_method(method: str) -> str:
    method = method.lower()
    aliases = {
        "centered": "finite_difference",
        "central": "finite_difference",
        "upwind": "upwind3",
        "upwind3": "upwind3",
        "third_order_upwind": "upwind3",
    }
    method = aliases.get(method, method)
    if method not in {"finite_difference", "spectral", "upwind3"}:
        raise ValueError(
            "spatial_derivative_method must be either 'finite_difference' or "
            f"'spectral' or 'upwind3', got {method!r}."
        )
    return method


def _validate_time_derivative_method(method: str) -> str:
    method = method.lower()
    if method not in {"central", "forward"}:
        raise ValueError(
            "time_derivative_method must be either 'central' or 'forward', "
            f"got {method!r}."
        )
    return method


def _spectral_first_derivative_periodic(
    field: np.ndarray,
    spacing: float,
    axis: int,
) -> np.ndarray:
    """Return a first derivative using a periodic Fourier spectral method."""
    count = field.shape[axis]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    reshape = [1] * field.ndim
    reshape[axis] = count
    multiplier = (1.0j * wave_numbers).reshape(reshape)

    field_hat = np.fft.fft(field, axis=axis)
    derivative_hat = multiplier * field_hat
    return np.fft.ifft(derivative_hat, axis=axis).real


def _spectral_second_derivative_periodic(
    field: np.ndarray,
    spacing: float,
    axis: int,
) -> np.ndarray:
    """Return a second derivative using a periodic Fourier spectral method."""
    count = field.shape[axis]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    reshape = [1] * field.ndim
    reshape[axis] = count
    multiplier = (-(wave_numbers**2)).reshape(reshape)

    field_hat = np.fft.fft(field, axis=axis)
    derivative_hat = multiplier * field_hat
    return np.fft.ifft(derivative_hat, axis=axis).real


def central_difference(
    field: np.ndarray,
    spacing: float,
    axis: int,
    periodic: bool = True,
) -> np.ndarray:
    """Return a first-order derivative using a centered stencil."""
    if periodic:
        return (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (
            2.0 * spacing
        )

    derivative = np.empty_like(field, dtype=float)
    interior = [slice(None)] * field.ndim
    interior[axis] = slice(1, -1)
    left = [slice(None)] * field.ndim
    left[axis] = slice(0, -2)
    right = [slice(None)] * field.ndim
    right[axis] = slice(2, None)
    derivative[tuple(interior)] = (
        field[tuple(right)] - field[tuple(left)]
    ) / (2.0 * spacing)

    start = [slice(None)] * field.ndim
    start[axis] = 0
    next_idx = [slice(None)] * field.ndim
    next_idx[axis] = 1
    derivative[tuple(start)] = (
        field[tuple(next_idx)] - field[tuple(start)]
    ) / spacing

    end = [slice(None)] * field.ndim
    end[axis] = -1
    prev_idx = [slice(None)] * field.ndim
    prev_idx[axis] = -2
    derivative[tuple(end)] = (field[tuple(end)] - field[tuple(prev_idx)]) / spacing
    return derivative


def second_difference(
    field: np.ndarray,
    spacing: float,
    axis: int,
    periodic: bool = True,
) -> np.ndarray:
    """Return a second derivative using a centered stencil."""
    if periodic:
        return (
            np.roll(field, -1, axis=axis) - 2.0 * field + np.roll(field, 1, axis=axis)
        ) / (spacing**2)

    derivative = np.empty_like(field, dtype=float)
    interior = [slice(None)] * field.ndim
    interior[axis] = slice(1, -1)
    left = [slice(None)] * field.ndim
    left[axis] = slice(0, -2)
    right = [slice(None)] * field.ndim
    right[axis] = slice(2, None)
    derivative[tuple(interior)] = (
        field[tuple(right)] - 2.0 * field[tuple(interior)] + field[tuple(left)]
    ) / (spacing**2)

    start = [slice(None)] * field.ndim
    start[axis] = 0
    next1 = [slice(None)] * field.ndim
    next1[axis] = 1
    next2 = [slice(None)] * field.ndim
    next2[axis] = 2
    derivative[tuple(start)] = (
        field[tuple(next2)] - 2.0 * field[tuple(next1)] + field[tuple(start)]
    ) / (spacing**2)

    end = [slice(None)] * field.ndim
    end[axis] = -1
    prev1 = [slice(None)] * field.ndim
    prev1[axis] = -2
    prev2 = [slice(None)] * field.ndim
    prev2[axis] = -3
    derivative[tuple(end)] = (
        field[tuple(end)] - 2.0 * field[tuple(prev1)] + field[tuple(prev2)]
    ) / (spacing**2)
    return derivative


def forward_difference(
    field: np.ndarray,
    spacing: float,
    axis: int,
    periodic: bool = False,
) -> np.ndarray:
    """Return a first derivative using a forward stencil."""
    if periodic:
        return (np.roll(field, -1, axis=axis) - field) / spacing

    derivative = np.empty_like(field, dtype=float)
    interior = [slice(None)] * field.ndim
    interior[axis] = slice(0, -1)
    right = [slice(None)] * field.ndim
    right[axis] = slice(1, None)
    derivative[tuple(interior)] = (
        field[tuple(right)] - field[tuple(interior)]
    ) / spacing

    end = [slice(None)] * field.ndim
    end[axis] = -1
    prev_idx = [slice(None)] * field.ndim
    prev_idx[axis] = -2
    derivative[tuple(end)] = (field[tuple(end)] - field[tuple(prev_idx)]) / spacing
    return derivative


def third_order_upwind_difference(
    field: np.ndarray,
    velocity_component: np.ndarray,
    spacing: float,
    axis: int,
    periodic: bool = True,
) -> np.ndarray:
    """Return the HPCC/solver-style third-order upwind first derivative.

    The derivative direction is selected pointwise from the sign of
    ``velocity_component`` using the same biased four-point stencil as the
    validation/checkq upwind3 pipeline. For non-periodic domains,
    boundary-adjacent points without a full stencil fall back to centered
    differences.
    """
    derivative = central_difference(field, spacing=spacing, axis=axis, periodic=periodic)

    velocity_mask = np.asarray(velocity_component, dtype=float)
    while velocity_mask.ndim < derivative.ndim:
        velocity_mask = velocity_mask[..., np.newaxis]

    positive_mask = velocity_mask >= 0.0
    negative_mask = velocity_mask < 0.0

    if periodic:
        positive_derivative = (
            2.0 * np.roll(field, -1, axis=axis)
            + 3.0 * field
            - 6.0 * np.roll(field, 1, axis=axis)
            + np.roll(field, 2, axis=axis)
        ) / (6.0 * spacing)
        negative_derivative = (
            -np.roll(field, -2, axis=axis)
            + 6.0 * np.roll(field, -1, axis=axis)
            - 3.0 * field
            - 2.0 * np.roll(field, 1, axis=axis)
        ) / (6.0 * spacing)
        derivative = np.where(positive_mask, positive_derivative, derivative)
        derivative = np.where(negative_mask, negative_derivative, derivative)
        return derivative

    axis_length = field.shape[axis]
    if axis_length < 3:
        return derivative

    center = [slice(None)] * field.ndim
    center[axis] = slice(2, -1)
    next1 = [slice(None)] * field.ndim
    next1[axis] = slice(3, None)
    prev1 = [slice(None)] * field.ndim
    prev1[axis] = slice(1, -2)
    prev2 = [slice(None)] * field.ndim
    prev2[axis] = slice(0, -3)
    positive_derivative = (
        2.0 * field[tuple(next1)]
        + 3.0 * field[tuple(center)]
        - 6.0 * field[tuple(prev1)]
        + field[tuple(prev2)]
    ) / (6.0 * spacing)
    derivative[tuple(center)] = np.where(
        positive_mask[tuple(center)],
        positive_derivative,
        derivative[tuple(center)],
    )

    center = [slice(None)] * field.ndim
    center[axis] = slice(1, -2)
    next1 = [slice(None)] * field.ndim
    next1[axis] = slice(2, -1)
    next2 = [slice(None)] * field.ndim
    next2[axis] = slice(3, None)
    prev1 = [slice(None)] * field.ndim
    prev1[axis] = slice(0, -3)
    negative_derivative = (
        -field[tuple(next2)]
        + 6.0 * field[tuple(next1)]
        - 3.0 * field[tuple(center)]
        - 2.0 * field[tuple(prev1)]
    ) / (6.0 * spacing)
    derivative[tuple(center)] = np.where(
        negative_mask[tuple(center)],
        negative_derivative,
        derivative[tuple(center)],
    )
    return derivative


def vector_time_derivative(
    vector_field: np.ndarray,
    dt: float,
    time_axis: int = 0,
    periodic: bool = False,
    time_derivative_method: str = "central",
) -> np.ndarray:
    """Compute the time derivative of a vector field."""
    time_derivative_method = _validate_time_derivative_method(time_derivative_method)
    if time_derivative_method == "forward":
        return forward_difference(
            vector_field,
            spacing=dt,
            axis=time_axis,
            periodic=periodic,
        )
    return central_difference(vector_field, spacing=dt, axis=time_axis, periodic=periodic)


def tensor_time_derivative(
    tensor_field: np.ndarray,
    dt: float,
    time_axis: int = 0,
    periodic: bool = False,
    time_derivative_method: str = "central",
) -> np.ndarray:
    """Compute the time derivative of a tensor-valued field."""
    time_derivative_method = _validate_time_derivative_method(time_derivative_method)
    if time_derivative_method == "forward":
        return forward_difference(
            tensor_field,
            spacing=dt,
            axis=time_axis,
            periodic=periodic,
        )
    return central_difference(tensor_field, spacing=dt, axis=time_axis, periodic=periodic)


def validate_velocity_field_3d(
    vector_field: np.ndarray,
    spatial_axes: Iterable[int],
) -> tuple[int, ...]:
    """Validate the convention v(t, x, y, z, component) for a 3D velocity field."""
    spatial_axes = tuple(spatial_axes)
    if len(spatial_axes) != 3:
        raise ValueError("A 3D velocity field requires exactly three spatial axes.")
    if vector_field.shape[-1] != 3:
        raise ValueError("The last axis of v must contain exactly three components.")
    return spatial_axes


def vector_gradient(
    vector_field: np.ndarray,
    spacings: Iterable[float],
    spatial_axes: Iterable[int],
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> list[np.ndarray]:
    """Compute partial derivatives of a vector field along each spatial axis."""
    spatial_derivative_method = _validate_spatial_derivative_method(
        spatial_derivative_method
    )
    spatial_axes = validate_velocity_field_3d(vector_field, spatial_axes)
    gradients: list[np.ndarray] = []
    for spacing, axis in zip(spacings, spatial_axes):
        if spatial_derivative_method == "spectral":
            if not periodic:
                raise ValueError(
                    "The spectral spatial derivative method requires periodic=True."
                )
            gradients.append(
                _spectral_first_derivative_periodic(
                    vector_field,
                    spacing=spacing,
                    axis=axis,
                )
            )
        else:
            gradients.append(
                central_difference(
                    vector_field,
                    spacing=spacing,
                    axis=axis,
                    periodic=periodic,
                )
            )
    return gradients


def vector_laplacian(
    vector_field: np.ndarray,
    spacings: Iterable[float],
    spatial_axes: Iterable[int],
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Compute the component-wise Laplacian of a vector field."""
    spatial_derivative_method = _validate_spatial_derivative_method(
        spatial_derivative_method
    )
    spatial_axes = validate_velocity_field_3d(vector_field, spatial_axes)
    result = np.zeros_like(vector_field, dtype=float)
    for spacing, axis in zip(spacings, spatial_axes):
        if spatial_derivative_method == "spectral":
            if not periodic:
                raise ValueError(
                    "The spectral spatial derivative method requires periodic=True."
                )
            result += _spectral_second_derivative_periodic(
                vector_field,
                spacing=spacing,
                axis=axis,
            )
        else:
            result += second_difference(
                vector_field,
                spacing=spacing,
                axis=axis,
                periodic=periodic,
            )
    return result


def tensor_laplacian(
    tensor_field: np.ndarray,
    spacings: Iterable[float],
    spatial_axes: Iterable[int],
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Compute the component-wise Laplacian of a 3x3 tensor field."""
    spatial_derivative_method = _validate_spatial_derivative_method(
        spatial_derivative_method
    )
    spatial_axes = tuple(spatial_axes)
    if len(spatial_axes) != 3:
        raise ValueError("A 3D tensor field requires exactly three spatial axes.")
    if tensor_field.shape[-2:] != (3, 3):
        raise ValueError("The last two axes of Q must each have length 3.")

    result = np.zeros_like(tensor_field, dtype=float)
    for spacing, axis in zip(spacings, spatial_axes):
        if spatial_derivative_method == "spectral":
            if not periodic:
                raise ValueError(
                    "The spectral spatial derivative method requires periodic=True."
                )
            result += _spectral_second_derivative_periodic(
                tensor_field,
                spacing=spacing,
                axis=axis,
            )
        else:
            result += second_difference(
                tensor_field,
                spacing=spacing,
                axis=axis,
                periodic=periodic,
            )
    return result


def tensor_divergence(
    tensor_field: np.ndarray,
    spacings: Iterable[float],
    spatial_axes: Iterable[int],
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Compute the divergence of a 3x3 tensor field."""
    spatial_derivative_method = _validate_spatial_derivative_method(
        spatial_derivative_method
    )
    spatial_axes = tuple(spatial_axes)
    if len(spatial_axes) != 3:
        raise ValueError("A 3D tensor field requires exactly three spatial axes.")
    if tensor_field.shape[-2:] != (3, 3):
        raise ValueError("The last two axes of Q must each have length 3.")

    result = np.zeros(tensor_field.shape[:-1], dtype=float)
    for column_index, (spacing, axis) in enumerate(zip(spacings, spatial_axes)):
        if spatial_derivative_method == "spectral":
            if not periodic:
                raise ValueError(
                    "The spectral spatial derivative method requires periodic=True."
                )
            result += _spectral_first_derivative_periodic(
                tensor_field[..., :, column_index],
                spacing=spacing,
                axis=axis,
            )
        else:
            result += central_difference(
                tensor_field[..., :, column_index],
                spacing=spacing,
                axis=axis,
                periodic=periodic,
            )
    return result


def tensor_gradient(
    tensor_field: np.ndarray,
    spacings: Iterable[float],
    spatial_axes: Iterable[int],
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Compute partial_a Q_bc for a 3x3 tensor field.

    The returned array has shape ``(..., 3, 3, 3)`` with the last three axes
    ordered as ``(derivative_component, row_component, column_component)``.
    """
    spatial_derivative_method = _validate_spatial_derivative_method(
        spatial_derivative_method
    )
    spatial_axes = tuple(spatial_axes)
    if len(spatial_axes) != 3:
        raise ValueError("A 3D tensor field requires exactly three spatial axes.")
    if tensor_field.shape[-2:] != (3, 3):
        raise ValueError("The last two axes of Q must each have length 3.")

    partials: list[np.ndarray] = []
    for spacing, axis in zip(spacings, spatial_axes):
        if spatial_derivative_method == "spectral":
            if not periodic:
                raise ValueError(
                    "The spectral spatial derivative method requires periodic=True."
                )
            partials.append(
                _spectral_first_derivative_periodic(
                    tensor_field,
                    spacing=spacing,
                    axis=axis,
                )
            )
        else:
            partials.append(
                central_difference(
                    tensor_field,
                    spacing=spacing,
                    axis=axis,
                    periodic=periodic,
                )
            )
    return np.stack(partials, axis=-3)


def vector_advection(
    advecting_field: np.ndarray,
    target_field: np.ndarray,
    spacings: Iterable[float],
    spatial_axes: Iterable[int],
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Compute (a dot grad) q for a 3D vector field a and a general field q."""
    spatial_derivative_method = _validate_spatial_derivative_method(
        spatial_derivative_method
    )
    spatial_axes = validate_velocity_field_3d(advecting_field, spatial_axes)

    result = np.zeros_like(target_field, dtype=float)
    for component_index, (spacing, axis) in enumerate(zip(spacings, spatial_axes)):
        velocity_component = advecting_field[..., component_index]
        if spatial_derivative_method == "spectral":
            if not periodic:
                raise ValueError(
                    "The spectral spatial derivative method requires periodic=True."
                )
            target_partial = _spectral_first_derivative_periodic(
                target_field,
                spacing=spacing,
                axis=axis,
            )
        elif spatial_derivative_method == "upwind3":
            target_partial = third_order_upwind_difference(
                target_field,
                velocity_component=velocity_component,
                spacing=spacing,
                axis=axis,
                periodic=periodic,
            )
        else:
            target_partial = central_difference(
                target_field,
                spacing=spacing,
                axis=axis,
                periodic=periodic,
            )
        while velocity_component.ndim < target_partial.ndim:
            velocity_component = velocity_component[..., np.newaxis]
        result += velocity_component * target_partial
    return result


def vector_material_derivative(
    vector_field: np.ndarray,
    spacings: Iterable[float],
    spatial_axes: Iterable[int],
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Compute the convective term (v dot grad) v component-wise."""
    return vector_advection(
        advecting_field=vector_field,
        target_field=vector_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
