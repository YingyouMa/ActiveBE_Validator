"""Pure two-dimensional Beris-Edwards validation API.

This namespace assumes both a two-dimensional spatial domain and a 2x2
symmetric-traceless Q tensor.
"""

from . import io, q_terms, validation, velocity_terms
from .io import discover_q_from_processed_npy
from .q_terms import reconstruct_q_tensor
from .validation import (
    fit_be_equation_velocity_local_weak_form,
    fit_be_equation_velocity_pointwise,
    fit_be_q_equation_local_weak_form,
    fit_be_q_equation_pointwise,
)

__all__ = [
    "discover_q_from_processed_npy",
    "fit_be_equation_velocity_local_weak_form",
    "fit_be_equation_velocity_pointwise",
    "fit_be_q_equation_local_weak_form",
    "fit_be_q_equation_pointwise",
    "io",
    "q_terms",
    "reconstruct_q_tensor",
    "validation",
    "velocity_terms",
]

