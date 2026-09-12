"""Two-dimensional differential helpers built from the shared stencils."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..operators import central_difference, second_difference, third_order_upwind_difference


def _validate_layout(
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
) -> tuple[tuple[float, float], tuple[int, int]]:
    spacings = tuple(float(value) for value in spacings)
    spatial_axes = tuple(int(axis) for axis in spatial_axes)
    if len(spacings) != 2 or len(spatial_axes) != 2:
        raise ValueError("Pure 2D fields require exactly two spacings and two spatial axes.")
    return spacings, spatial_axes


def _spectral_derivative(
    field: np.ndarray,
    *,
    spacing: float,
    axis: int,
    order: int,
) -> np.ndarray:
    count = field.shape[axis]
    wave_numbers = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    reshape = [1] * field.ndim
    reshape[axis] = count
    if order == 1:
        multiplier = (1.0j * wave_numbers).reshape(reshape)
    elif order == 2:
        multiplier = (-(wave_numbers**2)).reshape(reshape)
    else:
        raise ValueError("Only first and second spectral derivatives are supported.")
    transformed = np.fft.fft(field, axis=axis)
    return np.fft.ifft(multiplier * transformed, axis=axis).real


def first_derivative(
    field: np.ndarray,
    *,
    spacing: float,
    axis: int,
    periodic: bool,
    method: str,
) -> np.ndarray:
    method = method.lower()
    if method in {"finite_difference", "central", "centered"}:
        return central_difference(field, spacing=spacing, axis=axis, periodic=periodic)
    if method == "spectral":
        if not periodic:
            raise ValueError("Spectral derivatives require periodic=True.")
        return _spectral_derivative(field, spacing=spacing, axis=axis, order=1)
    raise ValueError(f"Unsupported derivative method {method!r}.")


def laplacian(
    field: np.ndarray,
    *,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
    periodic: bool = True,
    method: str = "finite_difference",
) -> np.ndarray:
    spacings, spatial_axes = _validate_layout(spacings, spatial_axes)
    result = np.zeros_like(field, dtype=float)
    for spacing, axis in zip(spacings, spatial_axes):
        if method == "spectral":
            if not periodic:
                raise ValueError("Spectral derivatives require periodic=True.")
            result += _spectral_derivative(field, spacing=spacing, axis=axis, order=2)
        else:
            result += second_difference(field, spacing=spacing, axis=axis, periodic=periodic)
    return result


def gradient(
    field: np.ndarray,
    *,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
    periodic: bool = True,
    method: str = "finite_difference",
) -> tuple[np.ndarray, np.ndarray]:
    spacings, spatial_axes = _validate_layout(spacings, spatial_axes)
    return tuple(
        first_derivative(
            field,
            spacing=spacing,
            axis=axis,
            periodic=periodic,
            method=method,
        )
        for spacing, axis in zip(spacings, spatial_axes)
    )


def vector_advection(
    velocity_field: np.ndarray,
    target_field: np.ndarray,
    *,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
    periodic: bool = True,
    method: str = "finite_difference",
) -> np.ndarray:
    spacings, spatial_axes = _validate_layout(spacings, spatial_axes)
    if velocity_field.shape[-1] != 2:
        raise ValueError("A pure 2D velocity field must end with exactly two components.")

    result = np.zeros_like(target_field, dtype=float)
    for component, (spacing, axis) in enumerate(zip(spacings, spatial_axes)):
        velocity_component = velocity_field[..., component]
        if method in {"upwind", "upwind3", "third_order_upwind"}:
            partial = third_order_upwind_difference(
                target_field,
                velocity_component=velocity_component,
                spacing=spacing,
                axis=axis,
                periodic=periodic,
            )
        else:
            partial = first_derivative(
                target_field,
                spacing=spacing,
                axis=axis,
                periodic=periodic,
                method=method,
            )
        while velocity_component.ndim < partial.ndim:
            velocity_component = velocity_component[..., np.newaxis]
        result += velocity_component * partial
    return result


def tensor_divergence(
    tensor_field: np.ndarray,
    *,
    spacings: Sequence[float],
    spatial_axes: Sequence[int],
    periodic: bool = True,
    method: str = "finite_difference",
) -> np.ndarray:
    if tensor_field.shape[-2:] != (2, 2):
        raise ValueError("A pure 2D tensor field must end with shape (2, 2).")
    spacings, spatial_axes = _validate_layout(spacings, spatial_axes)
    result = np.zeros(tensor_field.shape[:-1], dtype=float)
    for column, (spacing, axis) in enumerate(zip(spacings, spatial_axes)):
        result += first_derivative(
            tensor_field[..., :, column],
            spacing=spacing,
            axis=axis,
            periodic=periodic,
            method=method,
        )
    return result
