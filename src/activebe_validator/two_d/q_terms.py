"""Beris-Edwards Q-tensor terms for a pure two-dimensional nematic."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..operators import tensor_time_derivative
from ._differential import gradient, laplacian, vector_advection


def reconstruct_q_tensor(q_components: np.ndarray) -> np.ndarray:
    """Reconstruct a traceless 2x2 Q tensor from (..., Qxx, Qxy)."""
    q_components = np.asarray(q_components, dtype=float)
    if q_components.shape[-1] != 2:
        raise ValueError("2D Q components must end with (Qxx, Qxy).")
    qxx = q_components[..., 0]
    qxy = q_components[..., 1]
    result = np.empty(q_components.shape[:-1] + (2, 2), dtype=float)
    result[..., 0, 0] = qxx
    result[..., 0, 1] = qxy
    result[..., 1, 0] = qxy
    result[..., 1, 1] = -qxx
    return result


def extract_independent_q_components(q_tensor: np.ndarray) -> np.ndarray:
    """Return the two independent components (Qxx, Qxy)."""
    q_tensor = _validate_q(q_tensor)
    return np.stack((q_tensor[..., 0, 0], q_tensor[..., 0, 1]), axis=-1)


def _validate_q(q_tensor: np.ndarray) -> np.ndarray:
    q_tensor = np.asarray(q_tensor, dtype=float)
    if q_tensor.shape[-2:] != (2, 2):
        raise ValueError("A pure 2D Q tensor must end with shape (2, 2).")
    return q_tensor


def project_symmetric_traceless(tensor_field: np.ndarray) -> np.ndarray:
    tensor_field = np.asarray(tensor_field, dtype=float)
    if tensor_field.shape[-2:] != (2, 2):
        raise ValueError("Tensor field must end with shape (2, 2).")
    symmetric = 0.5 * (tensor_field + np.swapaxes(tensor_field, -1, -2))
    trace = np.einsum("...ii->...", symmetric)
    return symmetric - 0.5 * trace[..., None, None] * np.eye(2)


def bulk_linear_term(q_tensor: np.ndarray) -> np.ndarray:
    return _validate_q(q_tensor)


def bulk_quartic_term(q_tensor: np.ndarray) -> np.ndarray:
    """Return Tr(Q^2) Q, the derivative structure from the quartic invariant."""
    q_tensor = _validate_q(q_tensor)
    trace_q2 = np.einsum("...ij,...ji->...", q_tensor, q_tensor)
    return trace_q2[..., None, None] * q_tensor


def elastic_molecular_field_l1(
    q_tensor: np.ndarray,
    l1: float,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    return l1 * laplacian(
        _validate_q(q_tensor),
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        method=spatial_derivative_method,
    )


def elastic_free_energy_l1(
    q_tensor: np.ndarray,
    l1: float,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    q_grad = gradient(
        _validate_q(q_tensor),
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        method=spatial_derivative_method,
    )
    return 0.5 * l1 * sum(np.einsum("...ij,...ij->...", part, part) for part in q_grad)


def landau_de_gennes_free_energy_density(
    q_tensor: np.ndarray,
    *,
    a2: float,
    a4: float,
    kappa: float,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Return f = a2/2 Tr(Q^2) + a4/4 Tr(Q^2)^2 + kappa/2 |grad Q|^2.

    For a traceless 2x2 Q tensor, Tr(Q^3) vanishes identically, so there is no
    cubic invariant and no ``a3`` parameter in the pure-2D model.
    """
    q_tensor = _validate_q(q_tensor)
    trace_q2 = np.einsum("...ij,...ji->...", q_tensor, q_tensor)
    return (
        0.5 * a2 * trace_q2
        + 0.25 * a4 * trace_q2**2
        + elastic_free_energy_l1(
            q_tensor,
            l1=kappa,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
    )


def q_molecular_field(
    q_tensor: np.ndarray,
    *,
    a2: float,
    a4: float,
    kappa: float,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    q_tensor = _validate_q(q_tensor)
    return (
        -a2 * q_tensor
        - a4 * bulk_quartic_term(q_tensor)
        + elastic_molecular_field_l1(
            q_tensor,
            l1=kappa,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
    )


def q_time_derivative(
    q_tensor: np.ndarray,
    dt: float,
    time_axis: int = 0,
    periodic: bool = False,
    time_derivative_method: str = "central",
) -> np.ndarray:
    return tensor_time_derivative(
        _validate_q(q_tensor),
        dt=dt,
        time_axis=time_axis,
        periodic=periodic,
        time_derivative_method=time_derivative_method,
    )


def q_material_derivative(
    velocity_field: np.ndarray,
    q_tensor: np.ndarray,
    spacings: Sequence[float],
    spatial_axes: Sequence[int] = (0, 1),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    return vector_advection(
        velocity_field,
        _validate_q(q_tensor),
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        method=spatial_derivative_method,
    )


def q_omega_interaction(q_tensor: np.ndarray, omega_tensor: np.ndarray) -> np.ndarray:
    q_tensor = _validate_q(q_tensor)
    omega_tensor = np.asarray(omega_tensor, dtype=float)
    return np.einsum("...ik,...kj->...ij", omega_tensor, q_tensor) - np.einsum(
        "...ik,...kj->...ij", q_tensor, omega_tensor
    )


def q_e_plain_interaction(e_tensor: np.ndarray) -> np.ndarray:
    return project_symmetric_traceless(e_tensor)


def q_e_interaction(q_tensor: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
    q_tensor = _validate_q(q_tensor)
    interaction = np.einsum("...ik,...kj->...ij", e_tensor, q_tensor) + np.einsum(
        "...ik,...kj->...ij", q_tensor, e_tensor
    )
    return project_symmetric_traceless(interaction)


def q_colon_e_q_interaction(q_tensor: np.ndarray, e_tensor: np.ndarray) -> np.ndarray:
    q_tensor = _validate_q(q_tensor)
    contraction = np.einsum("...ij,...ij->...", q_tensor, e_tensor)
    return q_tensor * contraction[..., None, None]


def q_flow_alignment(
    q_tensor: np.ndarray,
    e_tensor: np.ndarray,
    omega_tensor: np.ndarray,
    lambda_flow_alignment: float = 1.0,
) -> np.ndarray:
    """Return the pure-2D Beris-Edwards flow-alignment tensor.

    The isotropic shift is I/2 rather than the I/3 used by a 3D Q tensor.
    """
    q_tensor = _validate_q(q_tensor)
    shifted_q = q_tensor + 0.5 * np.eye(2)
    lambda_e = lambda_flow_alignment * np.asarray(e_tensor, dtype=float)
    omega_tensor = np.asarray(omega_tensor, dtype=float)
    left = np.einsum("...ik,...kj->...ij", lambda_e + omega_tensor, shifted_q)
    right = np.einsum("...ik,...kj->...ij", shifted_q, lambda_e - omega_tensor)
    contraction = np.einsum("...ij,...ij->...", q_tensor, e_tensor)
    alignment = left + right - (
        2.0 * lambda_flow_alignment * shifted_q * contraction[..., None, None]
    )
    return project_symmetric_traceless(alignment)


__all__ = [
    "bulk_linear_term",
    "bulk_quartic_term",
    "elastic_free_energy_l1",
    "elastic_molecular_field_l1",
    "extract_independent_q_components",
    "landau_de_gennes_free_energy_density",
    "project_symmetric_traceless",
    "q_colon_e_q_interaction",
    "q_e_interaction",
    "q_e_plain_interaction",
    "q_flow_alignment",
    "q_material_derivative",
    "q_molecular_field",
    "q_omega_interaction",
    "q_time_derivative",
    "reconstruct_q_tensor",
]
