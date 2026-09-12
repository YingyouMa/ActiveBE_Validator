"""Foundational discrete operators used by velocity and Q-term builders."""

from .differential import (
    central_difference,
    forward_difference,
    second_difference,
    tensor_divergence,
    tensor_gradient,
    tensor_laplacian,
    tensor_time_derivative,
    third_order_upwind_difference,
    validate_velocity_field_3d,
    vector_advection,
    vector_gradient,
    vector_laplacian,
    vector_material_derivative,
    vector_time_derivative,
)

__all__ = [
    "central_difference",
    "forward_difference",
    "second_difference",
    "tensor_divergence",
    "tensor_gradient",
    "tensor_laplacian",
    "tensor_time_derivative",
    "third_order_upwind_difference",
    "validate_velocity_field_3d",
    "vector_advection",
    "vector_gradient",
    "vector_laplacian",
    "vector_material_derivative",
    "vector_time_derivative",
]
