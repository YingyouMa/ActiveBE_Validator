"""Named velocity-field terms and manufactured velocity helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .operators import (
    tensor_divergence,
    tensor_gradient,
    vector_gradient,
    vector_laplacian,
    vector_material_derivative,
    vector_time_derivative,
)
from .q_terms import landau_de_gennes_free_energy_density, q_molecular_field


@dataclass(frozen=True)
class TestFieldSpec:
    """Parameterization for a periodic divergence-free Beltrami test field."""

    a: float
    b: float
    c: float
    wave_number: int = 1


@dataclass(frozen=True)
class GridSpec:
    dt: float
    dx: tuple[float, ...]
    time_axis: int = 0
    spatial_axes: tuple[int, ...] = (1, 2, 3)
    periodic: bool = True


@dataclass(frozen=True)
class VectorTerm:
    name: str
    description: str
    build: Callable[[dict[str, np.ndarray], GridSpec], np.ndarray]


def build_beltrami_test_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    spec: TestFieldSpec,
) -> np.ndarray:
    """Build a divergence-free periodic Beltrami-style test field."""
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    k = spec.wave_number

    phi_x = spec.a * np.sin(k * Z) + spec.c * np.cos(k * Y)
    phi_y = spec.b * np.sin(k * X) + spec.a * np.cos(k * Z)
    phi_z = spec.c * np.sin(k * Y) + spec.b * np.cos(k * X)
    return np.stack([phi_x, phi_y, phi_z], axis=-1)


def aligned_quadratic_velocity(vector_field: np.ndarray) -> np.ndarray:
    """Return the aligned quadratic velocity field |v| v."""
    speed = np.linalg.norm(vector_field, axis=-1, keepdims=True)
    return speed * np.asarray(vector_field, dtype=float)


def velocity_gradient_tensor(
    vector_field: np.ndarray,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Return the velocity-gradient tensor grad_v[..., i, j] = partial_j v_i."""
    partials = vector_gradient(
        vector_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    return np.stack(partials, axis=-1)


def velocity_shear_and_rotation_tensors(
    vector_field: np.ndarray,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> tuple[np.ndarray, np.ndarray]:
    """Return E and Omega from one shared velocity-gradient evaluation."""
    grad_v = velocity_gradient_tensor(
        vector_field,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    grad_v_t = np.swapaxes(grad_v, -1, -2)
    e_tensor = 0.5 * (grad_v + grad_v_t)
    omega_tensor = 0.5 * (grad_v - grad_v_t)
    return e_tensor, omega_tensor


def active_q_divergence_force(
    q_tensor: np.ndarray,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
) -> np.ndarray:
    """Return the active forcing feature div(Q)."""
    return tensor_divergence(
        q_tensor,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )


def passive_backflow_stress(
    q_tensor: np.ndarray,
    molecular_field: np.ndarray,
    *,
    free_energy_density: np.ndarray,
    flow_alignment_xi: float,
    elastic_kappa: float,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    include_distortion_stress: bool = True,
) -> np.ndarray:
    """Return the solver-style passive stress tensor ``sigma_passive``."""
    q_tensor = np.asarray(q_tensor, dtype=float)
    molecular_field = np.asarray(molecular_field, dtype=float)
    if q_tensor.shape[-2:] != (3, 3) or molecular_field.shape[-2:] != (3, 3):
        raise ValueError("Q and H must both end with shape (3, 3).")

    identity = np.eye(3, dtype=q_tensor.dtype)
    shifted_q = q_tensor + identity / 3.0
    q_contract_h = np.einsum("...ij,...ij->...", q_tensor, molecular_field)
    hq_t = np.einsum("...ac,...bc->...ab", molecular_field, q_tensor)
    qh_t = np.einsum("...ac,...bc->...ab", q_tensor, molecular_field)
    h_shifted_q_t = np.einsum("...ac,...bc->...ab", molecular_field, shifted_q)
    shifted_q_h_t = np.einsum("...ac,...bc->...ab", shifted_q, molecular_field)

    stress = -free_energy_density[..., np.newaxis, np.newaxis] * identity
    stress += (
        -2.0 * flow_alignment_xi * shifted_q * q_contract_h[..., np.newaxis, np.newaxis]
        + flow_alignment_xi * (h_shifted_q_t + shifted_q_h_t)
        + (hq_t - qh_t)
    )

    if include_distortion_stress:
        q_grad = tensor_gradient(
            q_tensor,
            spacings=spacings,
            spatial_axes=spatial_axes,
            periodic=periodic,
            spatial_derivative_method=spatial_derivative_method,
        )
        distortion_stress = elastic_kappa * np.einsum(
            "...iab,...jab->...ij",
            q_grad,
            q_grad,
        )
        stress = stress + distortion_stress

    return stress


def passive_backflow_force(
    q_tensor: np.ndarray,
    *,
    a2: float,
    a3: float,
    a4: float,
    kappa: float,
    flow_alignment_xi: float,
    spacings: tuple[float, float, float],
    spatial_axes: tuple[int, int, int] = (0, 1, 2),
    periodic: bool = True,
    spatial_derivative_method: str = "finite_difference",
    include_distortion_stress: bool = True,
) -> np.ndarray:
    """Return the solver-style backflow feature ``-div(sigma_passive)``."""
    molecular_field = q_molecular_field(
        q_tensor,
        a2=a2,
        a3=a3,
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
        a3=a3,
        a4=a4,
        kappa=kappa,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )
    passive_stress = passive_backflow_stress(
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
        passive_stress,
        spacings=spacings,
        spatial_axes=spatial_axes,
        periodic=periodic,
        spatial_derivative_method=spatial_derivative_method,
    )


def term_v(fields: dict[str, np.ndarray], _grid: GridSpec) -> np.ndarray:
    return fields["v"]


def term_dt_v(fields: dict[str, np.ndarray], grid: GridSpec) -> np.ndarray:
    return vector_time_derivative(
        fields["v"],
        dt=grid.dt,
        time_axis=grid.time_axis,
        periodic=False,
    )


def term_laplacian_v(fields: dict[str, np.ndarray], grid: GridSpec) -> np.ndarray:
    return vector_laplacian(
        fields["v"],
        spacings=grid.dx,
        spatial_axes=grid.spatial_axes,
        periodic=grid.periodic,
    )


def term_material_v(fields: dict[str, np.ndarray], grid: GridSpec) -> np.ndarray:
    return vector_material_derivative(
        fields["v"],
        spacings=grid.dx,
        spatial_axes=grid.spatial_axes,
        periodic=grid.periodic,
    )


VECTOR_TERM_REGISTRY: dict[str, VectorTerm] = {
    "v": VectorTerm(
        name="v",
        description="The 3D velocity field v(t, x, y, z) with three components.",
        build=term_v,
    ),
    "dt_v": VectorTerm(
        name="dt_v",
        description="Centered finite-difference approximation of partial v / partial t for a 3D velocity field.",
        build=term_dt_v,
    ),
    "laplacian_v": VectorTerm(
        name="laplacian_v",
        description="Component-wise Laplacian of a 3D velocity field over x, y, and z.",
        build=term_laplacian_v,
    ),
    "material_v": VectorTerm(
        name="material_v",
        description="Convective term (v dot grad) v for a 3D velocity field.",
        build=term_material_v,
    ),
}


def build_vector_terms(
    term_names: list[str],
    fields: dict[str, np.ndarray],
    grid: GridSpec,
) -> dict[str, np.ndarray]:
    """Build a selected set of named vector terms."""
    built_terms: dict[str, np.ndarray] = {}
    for term_name in term_names:
        built_terms[term_name] = VECTOR_TERM_REGISTRY[term_name].build(fields, grid)
    return built_terms
