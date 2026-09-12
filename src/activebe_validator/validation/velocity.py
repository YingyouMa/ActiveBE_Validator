"""Public velocity-equation validation API."""

from ..ns_checker import (
    DEFAULT_TEST_FIELD_SPECS,
    BEEquationVelocityFitResult,
    BELocalEquationVelocityFitResult,
    BEPointwiseVelocityFitResult,
    LocalPatchSpec,
    NSLocalWeakFormFitResult,
    NSPointwiseFitResult,
    NSProjectedFitResult,
    NSWeakFormFitResult,
    fit_be_equation_velocity,
    fit_be_equation_velocity_local_weak_form,
    fit_be_equation_velocity_pointwise,
    fit_ns_local_weak_form_coefficients,
    fit_ns_pointwise_coefficients,
    fit_ns_projected_coefficients,
    fit_ns_weak_form_coefficients,
)

__all__ = [
    "DEFAULT_TEST_FIELD_SPECS",
    "BEEquationVelocityFitResult",
    "BELocalEquationVelocityFitResult",
    "BEPointwiseVelocityFitResult",
    "LocalPatchSpec",
    "NSLocalWeakFormFitResult",
    "NSPointwiseFitResult",
    "NSProjectedFitResult",
    "NSWeakFormFitResult",
    "fit_be_equation_velocity",
    "fit_be_equation_velocity_local_weak_form",
    "fit_be_equation_velocity_pointwise",
    "fit_ns_local_weak_form_coefficients",
    "fit_ns_pointwise_coefficients",
    "fit_ns_projected_coefficients",
    "fit_ns_weak_form_coefficients",
]

