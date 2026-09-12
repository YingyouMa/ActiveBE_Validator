"""Backward-compatible aliases for velocity-term helpers."""

from .velocity_terms import (
    VECTOR_TERM_REGISTRY,
    GridSpec,
    VectorTerm,
    aligned_quadratic_velocity,
    build_beltrami_test_field,
    build_vector_terms,
    term_dt_v,
    term_laplacian_v,
    term_material_v,
    term_v,
    velocity_gradient_tensor,
    velocity_shear_and_rotation_tensors,
)

__all__ = [
    "VECTOR_TERM_REGISTRY",
    "GridSpec",
    "VectorTerm",
    "aligned_quadratic_velocity",
    "build_beltrami_test_field",
    "build_vector_terms",
    "term_dt_v",
    "term_laplacian_v",
    "term_material_v",
    "term_v",
    "velocity_gradient_tensor",
    "velocity_shear_and_rotation_tensors",
]
