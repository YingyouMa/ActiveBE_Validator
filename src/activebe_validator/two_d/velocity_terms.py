"""Velocity and stress terms for a pure two-dimensional active nematic."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..operators import vector_time_derivative
from ._differential import gradient, laplacian, tensor_divergence, vector_advection
from .q_terms import landau_de_gennes_free_energy_density, q_molecular_field


def _validate_velocity(vector_field: np.ndarray) -> np.ndarray:
    vector_field = np.asarray(vector_field, dtype=float)
    if vector_field.shape[-1] != 2:
        raise ValueError("A pure 2D velocity field must end with two components.")
    return vector_field


def aligned_quadratic_velocity(vector_field: np.ndarray) -> np.ndarray:
    vector_field = _validate_velocity(vector_field)
    speed = np.linalg.norm(vector_field, axis=-1, keepdims=True)
    return speed * vector_field


def velocity_gradient_tensor(
    vector_field: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    vector_field = _validate_velocity(vector_field)
    partial_x, partial_y = gradient(
        vector_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        method=spatial_derivative_method,
    )
    return np.stack((partial_x, partial_y), axis=-1)


def velocity_shear_and_rotation_tensors(
    vector_field: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> tuple[np.ndarray, np.ndarray]:
    grad_v = velocity_gradient_tensor(
        vector_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    grad_v_t = np.swapaxes(grad_v, -1, -2)
    return 0.5 * (grad_v + grad_v_t), 0.5 * (grad_v - grad_v_t)


def velocity_material_derivative(
    vector_field: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    vector_field = _validate_velocity(vector_field)
    return vector_advection(
        vector_field,
        vector_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        method=spatial_derivative_method,
    )


def velocity_laplacian(
    vector_field: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    return laplacian(
        _validate_velocity(vector_field),
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        method=spatial_derivative_method,
    )


def velocity_time_derivative_2d(
    vector_field: np.ndarray,
    dt: float,
    time_axis: int = 0,
    time_derivative_method: str = "central",
) -> np.ndarray:
    return vector_time_derivative(
        _validate_velocity(vector_field),
        dt=dt,
        time_axis=time_axis,
        periodic=False,
        time_derivative_method=time_derivative_method,
    )


def active_q_divergence_force(
    q_tensor: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    return tensor_divergence(
        q_tensor,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        method=spatial_derivative_method,
    )


def passive_backflow_stress(
    q_tensor: np.ndarray,
    molecular_field: np.ndarray,
    *,
    free_energy_density: np.ndarray,
    flow_alignment_xi: float,
    elastic_kappa: float,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    include_distortion_stress: bool = True,
) -> np.ndarray:
    q_tensor = np.asarray(q_tensor, dtype=float)
    molecular_field = np.asarray(molecular_field, dtype=float)
    if q_tensor.shape[-2:] != (2, 2) or molecular_field.shape[-2:] != (2, 2):
        raise ValueError("Q and H must both end with shape (2, 2).")

    identity = np.eye(2, dtype=q_tensor.dtype)
    shifted_q = q_tensor + 0.5 * identity
    q_contract_h = np.einsum("...ij,...ij->...", q_tensor, molecular_field)
    hq_t = np.einsum("...ac,...bc->...ab", molecular_field, q_tensor)
    qh_t = np.einsum("...ac,...bc->...ab", q_tensor, molecular_field)
    h_shifted_q_t = np.einsum("...ac,...bc->...ab", molecular_field, shifted_q)
    shifted_q_h_t = np.einsum("...ac,...bc->...ab", shifted_q, molecular_field)

    stress = -free_energy_density[..., None, None] * identity
    stress += (
        -2.0 * flow_alignment_xi * shifted_q * q_contract_h[..., None, None]
        + flow_alignment_xi * (h_shifted_q_t + shifted_q_h_t)
        + (hq_t - qh_t)
    )

    if include_distortion_stress:
        q_grad = gradient(
            q_tensor,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            method=spatial_derivative_method,
        )
        distortion = np.empty(q_tensor.shape[:-2] + (2, 2), dtype=float)
        for i in range(2):
            for j in range(2):
                distortion[..., i, j] = elastic_kappa * np.einsum(
                    "...ab,...ab->...", q_grad[i], q_grad[j]
                )
        stress += distortion
    return stress


def passive_backflow_force(
    q_tensor: np.ndarray,
    *,
    a2: float,
    a4: float,
    kappa: float,
    flow_alignment_xi: float,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    include_distortion_stress: bool = True,
) -> np.ndarray:
    molecular_field = q_molecular_field(
        q_tensor,
        a2=a2,
        a4=a4,
        kappa=kappa,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    free_energy_density = landau_de_gennes_free_energy_density(
        q_tensor,
        a2=a2,
        a4=a4,
        kappa=kappa,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    stress = passive_backflow_stress(
        q_tensor,
        molecular_field,
        free_energy_density=free_energy_density,
        flow_alignment_xi=flow_alignment_xi,
        elastic_kappa=kappa,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
        include_distortion_stress=include_distortion_stress,
    )
    return -tensor_divergence(
        stress,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        method=spatial_derivative_method,
    )


__all__ = [
    "active_q_divergence_force",
    "aligned_quadratic_velocity",
    "passive_backflow_force",
    "passive_backflow_stress",
    "velocity_gradient_tensor",
    "velocity_laplacian",
    "velocity_material_derivative",
    "velocity_shear_and_rotation_tensors",
    "velocity_time_derivative_2d",
]
