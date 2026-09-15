"""Cross-check ActiveBE Validator's pure-2D terms against one PSANC2D GPU step.

This is a development/verification script rather than part of the public API.
The PSANC2D result arrays are supplied as base64-encoded raw float64 buffers so
the comparison can be run without coupling this repository to HPCC.
"""

from __future__ import annotations

import argparse
import base64

import numpy as np

from activebe_validator.two_d.q_terms import (
    extract_independent_q_components,
    q_colon_e_q_interaction,
    q_e_plain_interaction,
    q_material_derivative,
    q_omega_interaction,
    reconstruct_q_tensor,
)
from activebe_validator.two_d.velocity_terms import (
    active_q_divergence_force,
    velocity_shear_and_rotation_tensors,
)


NX = 12
NY = 14
LX = 7.5
LY = 9.1
DT = 0.013
ETA = 1.23
KAPPA = 0.71
A2 = -0.041
A4 = 2.4
ALPHA = -0.27
GAMMA = 0.84
LAMBDA1 = -0.37
LAMBDA3 = 0.29
LAMBDA_R = -0.63
FRICTION = 0.17


def initial_components() -> np.ndarray:
    i = np.arange(NX)[:, None]
    j = np.arange(NY)[None, :]
    qxx = (
        0.08 * np.cos(2 * np.pi * i / NX)
        + 0.031 * np.sin(4 * np.pi * j / NY)
        + 0.017 * np.cos(2 * np.pi * (i / NX + j / NY))
    )
    qxy = (
        0.063 * np.sin(2 * np.pi * i / NX - 2 * np.pi * j / NY)
        + 0.022 * np.cos(4 * np.pi * i / NX)
    )
    return np.stack((qxx, qxy), axis=-1)


def stokes_velocity(q_tensor: np.ndarray) -> np.ndarray:
    spacings = (LX / NX, LY / NY)
    divergence_q = active_q_divergence_force(
        q_tensor,
        spacings=spacings,
        spatial_axes=(0, 1),
        periodic=True,
        spatial_derivative_method="spectral",
    )
    force = ALPHA * divergence_q
    force_hat = np.fft.fft2(force, axes=(0, 1))
    kx = 2 * np.pi * np.fft.fftfreq(NX, d=spacings[0])[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(NY, d=spacings[1])[None, :]
    k2 = kx**2 + ky**2
    safe_k2 = np.where(k2 > 0, k2, 1.0)
    k_dot_f = kx * force_hat[..., 0] + ky * force_hat[..., 1]
    projected_x = force_hat[..., 0] - kx * k_dot_f / safe_k2
    projected_y = force_hat[..., 1] - ky * k_dot_f / safe_k2
    denominator = ETA * k2 + FRICTION
    velocity_hat = np.zeros_like(force_hat)
    nonzero = k2 > 0
    velocity_hat[..., 0][nonzero] = projected_x[nonzero] / denominator[nonzero]
    velocity_hat[..., 1][nonzero] = projected_y[nonzero] / denominator[nonzero]
    return np.fft.ifft2(velocity_hat, axes=(0, 1)).real


def validator_step() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    spacings = (LX / NX, LY / NY)
    q_components = initial_components()
    q_tensor = reconstruct_q_tensor(q_components)
    velocity = stokes_velocity(q_tensor)
    e_tensor, omega_tensor = velocity_shear_and_rotation_tensors(
        velocity,
        spacings=spacings,
        spatial_axes=(0, 1),
        periodic=True,
        spatial_derivative_method="spectral",
    )

    advection = q_material_derivative(
        velocity,
        q_tensor,
        spacings=spacings,
        spatial_axes=(0, 1),
        periodic=True,
        spatial_derivative_method="spectral",
    )
    e_plain = q_e_plain_interaction(e_tensor)
    q_colon_e_q = q_colon_e_q_interaction(q_tensor, e_tensor)
    omega_q = q_omega_interaction(q_tensor, omega_tensor)
    trace_q2 = np.einsum("...ij,...ji->...", q_tensor, q_tensor)
    bulk_quartic = trace_q2[..., None, None] * q_tensor

    # PSANC2D uses lambda_R * (Q Omega - Omega Q), whereas the validator's
    # q_omega_interaction basis is (Omega Q - Q Omega); hence the minus sign.
    explicit_rhs = (
        -advection
        + LAMBDA1 * e_plain
        + LAMBDA3 * q_colon_e_q
        - LAMBDA_R * omega_q
        - GAMMA * A4 * bulk_quartic
    )
    rhs_components = extract_independent_q_components(explicit_rhs)

    kx = 2 * np.pi * np.fft.fftfreq(NX, d=spacings[0])[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(NY, d=spacings[1])[None, :]
    k2 = kx**2 + ky**2
    q_hat = np.fft.fft2(q_components, axes=(0, 1))
    rhs_hat = np.fft.fft2(rhs_components, axes=(0, 1))
    denominator = 1.0 + DT * GAMMA * (A2 + KAPPA * k2)
    q_next_hat = (q_hat + DT * rhs_hat) / denominator[..., None]
    q_next = np.fft.ifft2(q_next_hat, axes=(0, 1)).real

    diagnostics = {
        "max_abs_advection": float(np.max(np.abs(advection))),
        "max_abs_e_plain": float(np.max(np.abs(e_plain))),
        "max_abs_q_colon_e_q": float(np.max(np.abs(q_colon_e_q))),
        "max_abs_omega_q": float(np.max(np.abs(omega_q))),
        "max_abs_bulk_quartic": float(np.max(np.abs(bulk_quartic))),
    }
    return q_next, velocity, diagnostics


def decode_psanc_buffer(encoded: str, components: int) -> np.ndarray:
    raw = base64.b64decode(encoded)
    expected_size = components * NX * NY
    data = np.frombuffer(raw, dtype="<f8")
    if data.size != expected_size:
        raise ValueError(f"Expected {expected_size} float64 values, got {data.size}.")
    # PSANC raw output is component-major: (component, Nx, Ny).
    return np.moveaxis(data.reshape(components, NX, NY), 0, -1)


def error_report(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    difference = actual - expected
    max_abs = float(np.max(np.abs(difference)))
    rms = float(np.sqrt(np.mean(difference**2)))
    scale = float(np.max(np.abs(expected)))
    relative = max_abs / scale if scale else max_abs
    print(f"{name}: max_abs={max_abs:.17e} rms={rms:.17e} rel={relative:.17e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-next-base64", required=True)
    parser.add_argument("--velocity-base64", required=True)
    args = parser.parse_args()

    q_expected = decode_psanc_buffer(args.q_next_base64, 2)
    u_expected = decode_psanc_buffer(args.velocity_base64, 2)
    q_actual, u_actual, diagnostics = validator_step()
    error_report("velocity", u_actual, u_expected)
    error_report("q_next", q_actual, q_expected)
    for name, value in diagnostics.items():
        print(f"{name}={value:.17e}")


if __name__ == "__main__":
    main()
