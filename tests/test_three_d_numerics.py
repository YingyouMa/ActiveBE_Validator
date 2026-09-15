import numpy as np

from activebe_validator.ns_checker import (
    fit_be_equation_velocity_local_weak_form,
    fit_be_equation_velocity_pointwise,
)
from activebe_validator.operators import vector_laplacian, vector_material_derivative
from activebe_validator.q_checker import (
    fit_be_q_equation_local_weak_form,
    fit_be_q_equation_pointwise,
)
from activebe_validator.q_terms import (
    bulk_cubic_term,
    bulk_linear_term,
    bulk_quadratic_term,
    elastic_molecular_field_l1,
    project_symmetric_traceless,
    q_e_interaction,
    q_material_derivative,
    q_omega_interaction,
)
from activebe_validator.velocity_terms import (
    active_q_divergence_force,
    velocity_shear_and_rotation_tensors,
)


def _smooth_3d_fields(nx=10, ny=9, nz=8, lx=6.0, ly=5.0, lz=4.0):
    x = np.arange(nx)[:, None, None] * lx / nx
    y = np.arange(ny)[None, :, None] * ly / ny
    z = np.arange(nz)[None, None, :] * lz / nz

    raw = np.zeros((nx, ny, nz, 3, 3), dtype=float)
    raw[..., 0, 0] = 0.10 * np.cos(2 * np.pi * x / lx) + 0.025 * np.sin(2 * np.pi * y / ly)
    raw[..., 1, 1] = 0.07 * np.sin(2 * np.pi * y / ly) + 0.018 * np.cos(4 * np.pi * z / lz)
    raw[..., 0, 1] = raw[..., 1, 0] = 0.045 * np.sin(
        2 * np.pi * x / lx - 2 * np.pi * y / ly
    )
    raw[..., 0, 2] = raw[..., 2, 0] = 0.032 * np.cos(
        2 * np.pi * x / lx + 2 * np.pi * z / lz
    )
    raw[..., 1, 2] = raw[..., 2, 1] = 0.028 * np.sin(
        2 * np.pi * y / ly - 2 * np.pi * z / lz
    )
    q = project_symmetric_traceless(raw)

    u = np.empty((nx, ny, nz, 3), dtype=float)
    u[..., 0] = 0.12 * np.sin(2 * np.pi * y / ly) + 0.035 * np.cos(2 * np.pi * z / lz)
    u[..., 1] = 0.09 * np.sin(2 * np.pi * z / lz) + 0.027 * np.cos(2 * np.pi * x / lx)
    u[..., 2] = 0.08 * np.sin(2 * np.pi * x / lx) + 0.031 * np.cos(2 * np.pi * y / ly)
    return q, u, (lx / nx, ly / ny, lz / nz), (lx, ly, lz)


def test_3d_spectral_laplacian_matches_analytic_fourier_mode():
    nx, ny, nz = 11, 10, 9
    lx, ly, lz = 7.0, 6.0, 5.0
    x = np.arange(nx)[:, None, None] * lx / nx
    y = np.arange(ny)[None, :, None] * ly / ny
    z = np.arange(nz)[None, None, :] * lz / nz
    phase = 2 * np.pi * x / lx + 4 * np.pi * y / ly - 2 * np.pi * z / lz

    amplitude = np.array(
        [[0.7, 0.2, -0.1], [0.2, -0.3, 0.15], [-0.1, 0.15, -0.4]], dtype=float
    )
    q = np.cos(phase)[..., None, None] * amplitude
    spacings = (lx / nx, ly / ny, lz / nz)
    lap_q = elastic_molecular_field_l1(
        q,
        l1=1.0,
        spacings=spacings,
        periodic=True,
        spatial_derivative_method="spectral",
    )
    k2 = (2 * np.pi / lx) ** 2 + (4 * np.pi / ly) ** 2 + (2 * np.pi / lz) ** 2
    np.testing.assert_allclose(lap_q, -k2 * q, rtol=2e-12, atol=2e-12)


def test_3d_active_q_divergence_matches_analytic_fourier_derivatives():
    q, _, spacings, lengths = _smooth_3d_fields()
    lx, ly, lz = lengths
    nx, ny, nz = q.shape[:3]
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=lx / nx)[:, None, None]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=ly / ny)[None, :, None]
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=lz / nz)[None, None, :]
    qh = np.fft.fftn(q, axes=(0, 1, 2))
    reference_h = 1j * (
        kx[..., None] * qh[..., :, 0]
        + ky[..., None] * qh[..., :, 1]
        + kz[..., None] * qh[..., :, 2]
    )
    reference = np.fft.ifftn(reference_h, axes=(0, 1, 2)).real
    measured = active_q_divergence_force(
        q,
        spacings=spacings,
        periodic=True,
        spatial_derivative_method="spectral",
    )
    np.testing.assert_allclose(measured, reference, rtol=3e-12, atol=3e-12)


_Q_COEFFICIENTS = {
    "material_q": -0.83,
    "omega_q": 0.37,
    "e_q": -0.22,
    "bulk_linear": 0.041,
    "bulk_quadratic": -0.31,
    "bulk_cubic": -0.74,
    "elastic_l1": 0.46,
}


def _exact_3d_q_trajectory(frame_count=6, dt=0.004):
    q, u0, spacings, _ = _smooth_3d_fields()
    q_frames = [q]
    u_frames = []
    for _ in range(frame_count):
        u_frames.append(u0)
    for _ in range(frame_count - 1):
        current = q_frames[-1]
        e, omega = velocity_shear_and_rotation_tensors(
            u0,
            spacings=spacings,
            periodic=True,
            spatial_derivative_method="spectral",
        )
        rhs = (
            _Q_COEFFICIENTS["material_q"]
            * q_material_derivative(
                u0,
                current,
                spacings=spacings,
                periodic=True,
                spatial_derivative_method="spectral",
            )
            + _Q_COEFFICIENTS["omega_q"] * q_omega_interaction(current, omega)
            + _Q_COEFFICIENTS["e_q"] * q_e_interaction(current, e)
            + _Q_COEFFICIENTS["bulk_linear"] * bulk_linear_term(current)
            + _Q_COEFFICIENTS["bulk_quadratic"] * bulk_quadratic_term(current)
            + _Q_COEFFICIENTS["bulk_cubic"] * bulk_cubic_term(current)
            + _Q_COEFFICIENTS["elastic_l1"]
            * elastic_molecular_field_l1(
                current,
                l1=1.0,
                spacings=spacings,
                periodic=True,
                spatial_derivative_method="spectral",
            )
        )
        q_frames.append(project_symmetric_traceless(current + dt * rhs))
    return np.stack(u_frames), np.stack(q_frames), spacings, dt


def _assert_q_coefficients(result, *, rtol, atol):
    recovered = {
        name.removeprefix("local_"): coefficient
        for name, coefficient in zip(result.feature_names, result.coefficients, strict=True)
    }
    for name, expected in _Q_COEFFICIENTS.items():
        np.testing.assert_allclose(recovered[name], expected, rtol=rtol, atol=atol)


def test_3d_pointwise_coefficient_recovery_on_exact_discrete_trajectory():
    u, q, spacings, dt = _exact_3d_q_trajectory()
    result = fit_be_q_equation_pointwise(
        u,
        q,
        dt,
        spacings,
        periodic=True,
        spatial_derivative_method="spectral",
        v_time_level="previous",
        q_time_level="previous",
        time_discretization="two_point",
        use_interior_time_only=False,
        include_terms=tuple(_Q_COEFFICIENTS),
        threshold=0.0,
        optimizer_alpha=0.0,
    )
    _assert_q_coefficients(result, rtol=3e-8, atol=3e-9)
    assert result.relative_residual < 2e-10


def test_3d_local_weak_form_coefficient_recovery_on_exact_discrete_trajectory():
    u, q, spacings, dt = _exact_3d_q_trajectory()
    result = fit_be_q_equation_local_weak_form(
        u,
        q,
        dt,
        spacings,
        patch_half_widths=(2, 2, 2),
        patch_strides=(3, 3, 3),
        periodic=True,
        spatial_derivative_method="spectral",
        v_time_level="previous",
        q_time_level="previous",
        time_discretization="two_point",
        use_interior_time_only=False,
        threshold=0.0,
        optimizer_alpha=0.0,
    )
    _assert_q_coefficients(result, rtol=2e-6, atol=2e-7)
    assert result.relative_residual < 2e-8


_VELOCITY_COEFFICIENTS = {
    "linear_velocity": -0.19,
    "material": -0.57,
    "laplacian": 0.34,
    "q_divergence": -0.28,
}


def _exact_3d_velocity_trajectory(frame_count=6, dt=1e-4):
    q, u, spacings, _ = _smooth_3d_fields(nx=12, ny=11, nz=10)
    velocity_frames = [u.copy()]
    q_frames = [q.copy()]
    for _ in range(frame_count - 1):
        current = velocity_frames[-1]
        rhs = (
            _VELOCITY_COEFFICIENTS["linear_velocity"] * current
            + _VELOCITY_COEFFICIENTS["material"]
            * vector_material_derivative(
                current,
                spacings=spacings,
                spatial_axes=(0, 1, 2),
                periodic=True,
                spatial_derivative_method="finite_difference",
            )
            + _VELOCITY_COEFFICIENTS["laplacian"]
            * vector_laplacian(
                current,
                spacings=spacings,
                spatial_axes=(0, 1, 2),
                periodic=True,
                spatial_derivative_method="finite_difference",
            )
            + _VELOCITY_COEFFICIENTS["q_divergence"]
            * active_q_divergence_force(
                q,
                spacings=spacings,
                spatial_axes=(0, 1, 2),
                periodic=True,
                spatial_derivative_method="finite_difference",
            )
        )
        velocity_frames.append(current + dt * rhs)
        q_frames.append(q.copy())
    return np.stack(velocity_frames), np.stack(q_frames), spacings, dt


def test_3d_velocity_pointwise_coefficient_recovery_on_exact_discrete_trajectory():
    velocity, q, spacings, dt = _exact_3d_velocity_trajectory()
    pressure = np.zeros(velocity.shape[:-1], dtype=float)
    result = fit_be_equation_velocity_pointwise(
        velocity,
        pressure,
        1.0,
        q,
        dt,
        spacings,
        a2=0.0,
        a3=0.0,
        a4=0.0,
        kappa=0.0,
        periodic=True,
        spatial_derivative_method="finite_difference",
        time_derivative_method="forward",
        use_interior_time_only=True,
        include_terms=tuple(_VELOCITY_COEFFICIENTS),
        threshold=0.0,
        alpha=0.0,
    )
    recovered = dict(zip(result.feature_names, result.coefficients, strict=True))
    for name, expected in _VELOCITY_COEFFICIENTS.items():
        np.testing.assert_allclose(recovered[name], expected, rtol=2e-8, atol=2e-9)
    assert result.relative_residual < 2e-10


def test_3d_velocity_local_weak_form_coefficient_recovery_on_exact_discrete_trajectory():
    velocity, q, spacings, dt = _exact_3d_velocity_trajectory()
    result = fit_be_equation_velocity_local_weak_form(
        velocity,
        q,
        dt,
        spacings,
        viscosity=_VELOCITY_COEFFICIENTS["laplacian"],
        patch_half_widths=(2, 2, 2),
        patch_strides=(3, 3, 3),
        periodic=True,
        spatial_derivative_method="finite_difference",
        time_derivative_method="forward",
        num_test_fields_per_patch=3,
        use_interior_time_only=True,
        threshold=0.0,
        optimizer_alpha=0.0,
    )
    np.testing.assert_allclose(result.dt_coefficient, 1.0, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        result.linear_velocity_coefficient,
        _VELOCITY_COEFFICIENTS["linear_velocity"],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result.material_coefficient,
        _VELOCITY_COEFFICIENTS["material"],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result.q_divergence_coefficient,
        _VELOCITY_COEFFICIENTS["q_divergence"],
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result.viscosity_coefficient,
        _VELOCITY_COEFFICIENTS["laplacian"],
        rtol=0,
        atol=0,
    )
    assert result.relative_residual < 2e-8
