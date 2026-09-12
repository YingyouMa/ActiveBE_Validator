"""Named Q-tensor terms for Beris-Edwards and LdG-style checks."""

from __future__ import annotations

import numpy as np

from .operators import tensor_gradient, tensor_laplacian, tensor_time_derivative, vector_advection


def project_symmetric_traceless(tensor_field: np.ndarray) -> np.ndarray:
    """Project a 3x3 tensor field onto its symmetric traceless part."""
    tensor_field = np.asarray(tensor_field, dtype=float)
    if tensor_field.shape[-2:] != (3, 3):
        raise ValueError("Tensor field must have shape (..., 3, 3).")

    symmetric_part = 0.5 * (tensor_field + np.swapaxes(tensor_field, -1, -2))
    trace = np.einsum("...ii->...", symmetric_part)
    identity = np.eye(3, dtype=symmetric_part.dtype)
    return symmetric_part - trace[..., np.newaxis, np.newaxis] * identity / 3.0


def bulk_linear_term(q_tensor: np.ndarray) -> np.ndarray:
    """Return the linear bulk tensor term Q."""
    q_tensor = np.asarray(q_tensor, dtype=float)
    if q_tensor.shape[-2:] != (3, 3):
        raise ValueError("Q must have shape (..., 3, 3).")
    return q_tensor


def bulk_quadratic_term(q_tensor: np.ndarray) -> np.ndarray:
    """Return Q^2 - I Tr(Q^2) / 3."""
    q_tensor = np.asarray(q_tensor, dtype=float)
    if q_tensor.shape[-2:] != (3, 3):
        raise ValueError("Q must have shape (..., 3, 3).")

    trace_q2 = np.einsum("...ij,...ji->...", q_tensor, q_tensor)
    q_squared = np.einsum("...ik,...kj->...ij", q_tensor, q_tensor)
    identity = np.eye(3, dtype=q_tensor.dtype)
    return q_squared - trace_q2[..., np.newaxis, np.newaxis] * identity / 3.0


def bulk_cubic_term(q_tensor: np.ndarray) -> np.ndarray:
    """Return Tr(Q^2) Q."""
    q_tensor = np.asarray(q_tensor, dtype=float)
    if q_tensor.shape[-2:] != (3, 3):
        raise ValueError("Q must have shape (..., 3, 3).")

    trace_q2 = np.einsum("...ij,...ji->...", q_tensor, q_tensor)
    return trace_q2[..., np.newaxis, np.newaxis] * q_tensor


def bulk_free_energy(
    q_tensor: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    """Return the positive-sign LdG bulk contribution for a 3x3 Q-tensor field.

    The returned tensor is

        a Q + b (Q^2 - I Tr(Q^2) / 3) + c Tr(Q^2) Q

    with ``Q`` stored using the convention ``(..., 3, 3)``.
    """
    q_tensor = np.asarray(q_tensor, dtype=float)
    if q_tensor.shape[-2:] != (3, 3):
        raise ValueError("Q must have shape (..., 3, 3).")

    return a * bulk_linear_term(q_tensor) + b * bulk_quadratic_term(
        q_tensor
    ) + c * bulk_cubic_term(q_tensor)


def elastic_free_energy_l1(
    q_tensor: np.ndarray,
    l1: float,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Return the one-constant L1 elastic free-energy density.

    This evaluates

        (L1 / 2) * (partial_a Q_bc) * (partial_a Q_bc)

    for a tensor field with layout ``(..., 3, 3)``.
    """
    q_grad = tensor_gradient(
        q_tensor,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    return 0.5 * l1 * np.einsum("...abc,...abc->...", q_grad, q_grad)


def landau_de_gennes_free_energy_density(
    q_tensor: np.ndarray,
    *,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Return the solver-style LdG free-energy density ``fed``."""
    q_tensor = np.asarray(q_tensor, dtype=float)
    if q_tensor.shape[-2:] != (3, 3):
        raise ValueError("Q must have shape (..., 3, 3).")

    trace_q2 = np.einsum("...ij,...ji->...", q_tensor, q_tensor)
    q_squared = np.einsum("...ik,...kj->...ij", q_tensor, q_tensor)
    trace_q3 = np.einsum("...ij,...ji->...", q_squared, q_tensor)
    return (
        0.5 * a2 * trace_q2
        + (a3 / 3.0) * trace_q3
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


def elastic_molecular_field_l1(
    q_tensor: np.ndarray,
    l1: float,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Return the positive-sign L1 tensor contribution L1 Delta Q."""
    return l1 * tensor_laplacian(
        q_tensor,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )


def q_molecular_field(
    q_tensor: np.ndarray,
    *,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Return the solver-style molecular field H(Q)."""
    return (
        -a2 * bulk_linear_term(q_tensor)
        - a3 * bulk_quadratic_term(q_tensor)
        - a4 * bulk_cubic_term(q_tensor)
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
    """Return partial_t Q."""
    return tensor_time_derivative(
        q_tensor,
        dt=dt,
        time_axis=time_axis,
        periodic=periodic,
        time_derivative_method=time_derivative_method,
    )


def q_material_derivative(
    velocity_field: np.ndarray,
    q_tensor: np.ndarray,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Return the advective tensor term (v dot grad) Q."""
    return vector_advection(
        advecting_field=velocity_field,
        target_field=q_tensor,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )


def q_omega_interaction(
    q_tensor: np.ndarray,
    omega_tensor: np.ndarray,
) -> np.ndarray:
    """Return the commutator interaction Omega Q - Q Omega."""
    omega_q = np.einsum("...ik,...kj->...ij", omega_tensor, q_tensor)
    q_omega = np.einsum("...ik,...kj->...ij", q_tensor, omega_tensor)
    return omega_q - q_omega


def q_e_interaction(
    q_tensor: np.ndarray,
    e_tensor: np.ndarray,
) -> np.ndarray:
    """Return the symmetric-traceless projection of E Q + Q E."""
    e_q = np.einsum("...ik,...kj->...ij", e_tensor, q_tensor)
    q_e = np.einsum("...ik,...kj->...ij", q_tensor, e_tensor)
    return project_symmetric_traceless(e_q + q_e)


def q_e_plain_interaction(e_tensor: np.ndarray) -> np.ndarray:
    """Return the symmetric-traceless projection of E."""
    return project_symmetric_traceless(e_tensor)


def q_colon_e_q_interaction(
    q_tensor: np.ndarray,
    e_tensor: np.ndarray,
) -> np.ndarray:
    """Return the symmetric-traceless projection of ``(Q:E) Q``."""
    q_tensor = np.asarray(q_tensor, dtype=float)
    e_tensor = np.asarray(e_tensor, dtype=float)
    if q_tensor.shape[-2:] != (3, 3):
        raise ValueError("Q must have shape (..., 3, 3).")
    if e_tensor.shape[-2:] != (3, 3):
        raise ValueError("E must have shape (..., 3, 3).")

    q_contract_e = np.einsum("...ij,...ij->...", q_tensor, e_tensor)
    return project_symmetric_traceless(
        q_tensor * q_contract_e[..., np.newaxis, np.newaxis]
    )


def q_flow_alignment(
    q_tensor: np.ndarray,
    e_tensor: np.ndarray,
    omega_tensor: np.ndarray,
    lambda_flow_alignment: float = 1.0,
) -> np.ndarray:
    """Return the Beris-Edwards flow-alignment tensor S(W, Q).

    This evaluates

        (lambda E + Omega)(Q + I/3) + (Q + I/3)(lambda E - Omega)
        - 2 lambda (Q + I/3) (Q:E)

    and projects the result to the symmetric-traceless subspace. For a
    divergence-free velocity field this projection only removes small numerical
    trace errors.
    """
    q_tensor = np.asarray(q_tensor, dtype=float)
    e_tensor = np.asarray(e_tensor, dtype=float)
    omega_tensor = np.asarray(omega_tensor, dtype=float)
    if q_tensor.shape[-2:] != (3, 3):
        raise ValueError("Q must have shape (..., 3, 3).")
    if e_tensor.shape[-2:] != (3, 3) or omega_tensor.shape[-2:] != (3, 3):
        raise ValueError("E and Omega must have shape (..., 3, 3).")

    identity = np.eye(3, dtype=q_tensor.dtype)
    shifted_q = q_tensor + identity / 3.0
    lambda_e = lambda_flow_alignment * e_tensor

    left = np.einsum("...ik,...kj->...ij", lambda_e + omega_tensor, shifted_q)
    right = np.einsum("...ik,...kj->...ij", shifted_q, lambda_e - omega_tensor)
    q_contract_e = np.einsum("...ij,...ij->...", q_tensor, e_tensor)
    alignment = left + right - (
        2.0
        * lambda_flow_alignment
        * shifted_q
        * q_contract_e[..., np.newaxis, np.newaxis]
    )
    return project_symmetric_traceless(alignment)
