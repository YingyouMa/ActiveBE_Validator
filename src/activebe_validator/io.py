"""High-level processed-output discovery helpers."""

from .discovery import (
    InferredQMaterialParameters,
    PressureLeftVelocityDiscoveryResult,
    ProcessedOutputDiscoveryResult,
    ProcessedQDiscoveryResult,
    discover_from_processed_npy,
    discover_from_processed_npy_local_weak_form,
    discover_q_from_processed_npy,
)

__all__ = [
    "InferredQMaterialParameters",
    "PressureLeftVelocityDiscoveryResult",
    "ProcessedOutputDiscoveryResult",
    "ProcessedQDiscoveryResult",
    "discover_from_processed_npy",
    "discover_from_processed_npy_local_weak_form",
    "discover_q_from_processed_npy",
]
