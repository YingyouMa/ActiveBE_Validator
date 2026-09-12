"""Public velocity and stress term-building API."""

from ..velocity_terms import (
    TestFieldSpec,
    active_q_divergence_force,
    aligned_quadratic_velocity,
    build_beltrami_test_field,
    passive_backflow_force,
    passive_backflow_stress,
    velocity_gradient_tensor,
    velocity_shear_and_rotation_tensors,
)

__all__ = [
    "TestFieldSpec",
    "active_q_divergence_force",
    "aligned_quadratic_velocity",
    "build_beltrami_test_field",
    "passive_backflow_force",
    "passive_backflow_stress",
    "velocity_gradient_tensor",
    "velocity_shear_and_rotation_tensors",
]

