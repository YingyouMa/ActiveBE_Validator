import numpy as np

from activebe_validator.two_d.q_terms import (
    bulk_linear_term,
    bulk_quartic_term,
    elastic_molecular_field_l1,
    q_colon_e_q_interaction,
    q_e_plain_interaction,
    q_material_derivative,
    q_omega_interaction,
    reconstruct_q_tensor,
)
from activebe_validator.two_d.validation import (
    fit_be_equation_velocity_local_weak_form,
    fit_be_equation_velocity_pointwise,
    fit_be_q_equation_local_weak_form,
    fit_be_q_equation_pointwise,
)
from activebe_validator.two_d.velocity_terms import (
    active_q_divergence_force,
    velocity_laplacian,
    velocity_material_derivative,
    velocity_shear_and_rotation_tensors,
)


def _analytic_fields(nt=4, nx=18, ny=20, dt=0.01, lx=7.5, ly=9.1):
    x = np.arange(nx)[:, None] * lx / nx
    y = np.arange(ny)[None, :] * ly / ny
    qs = []
    us = []
    for n in range(nt):
        t = n * dt
        qxx = 0.08 * np.cos(2*np.pi*x/lx + .3*t) + 0.031*np.sin(4*np.pi*y/ly - .2*t)
        qxy = 0.063 * np.sin(2*np.pi*x/lx - 2*np.pi*y/ly + .17*t) + 0.022*np.cos(4*np.pi*x/lx)
        ux = .11*np.sin(2*np.pi*y/ly) + .04*np.cos(2*np.pi*x/lx + .11*t)
        uy = .09*np.sin(2*np.pi*x/lx) - .03*np.cos(2*np.pi*y/ly - .07*t)
        qs.append(reconstruct_q_tensor(np.stack((qxx, qxy), axis=-1)))
        us.append(np.stack((ux + np.zeros_like(qxx), uy + np.zeros_like(qxx)), axis=-1))
    return np.stack(us), np.stack(qs), (lx/nx, ly/ny)


def test_2d_spectral_basis_matches_direct_fourier_derivatives():
    u, q, spacings = _analytic_fields(nt=1)
    u, q = u[0], q[0]
    e, omega = velocity_shear_and_rotation_tensors(
        u, spacings, spatial_axes=(0, 1), periodic=True, spatial_derivative_method="spectral"
    )
    fields = (
        q_material_derivative(u, q, spacings, spatial_axes=(0, 1), spatial_derivative_method="spectral"),
        q_e_plain_interaction(e),
        q_colon_e_q_interaction(q, e),
        q_omega_interaction(q, omega),
        bulk_linear_term(q),
        bulk_quartic_term(q),
        elastic_molecular_field_l1(q, 1.0, spacings, spatial_axes=(0, 1), spatial_derivative_method="spectral"),
    )
    for field in fields:
        assert np.all(np.isfinite(field))
        assert np.max(np.abs(field[..., 0, 1] - field[..., 1, 0])) < 1e-13
        assert np.max(np.abs(field[..., 0, 0] + field[..., 1, 1])) < 1e-13


def _synthetic_exact_trajectory(nt=5, nx=24, ny=26, dt=2e-4):
    u0, q0, spacings = _analytic_fields(nt=1, nx=nx, ny=ny, dt=dt)
    q = q0[0]
    u = u0[0]
    coeff = {
        "material_q": -1.0,
        "omega_q": 0.41,
        "e_plain": -0.28,
        "q_colon_e_q": 0.19,
        "bulk_linear": 0.037,
        "bulk_quartic": -1.3,
        "elastic_l1": 0.52,
    }
    qs = [q.copy()]
    us = [u.copy()]
    for _ in range(nt - 1):
        e, omega = velocity_shear_and_rotation_tensors(
            u, spacings, spatial_axes=(0, 1), spatial_derivative_method="spectral"
        )
        basis = {
            "material_q": q_material_derivative(u, q, spacings, spatial_axes=(0, 1), spatial_derivative_method="spectral"),
            "omega_q": q_omega_interaction(q, omega),
            "e_plain": q_e_plain_interaction(e),
            "q_colon_e_q": q_colon_e_q_interaction(q, e),
            "bulk_linear": bulk_linear_term(q),
            "bulk_quartic": bulk_quartic_term(q),
            "elastic_l1": elastic_molecular_field_l1(q, 1.0, spacings, spatial_axes=(0, 1), spatial_derivative_method="spectral"),
        }
        rhs = sum(coeff[name] * basis[name] for name in coeff)
        q = q + dt * rhs
        qs.append(q.copy())
        us.append(u.copy())
    return np.stack(us), np.stack(qs), spacings, coeff


def test_2d_pointwise_coefficient_recovery_on_exact_discrete_trajectory():
    u, q, spacings, expected = _synthetic_exact_trajectory()
    names = tuple(expected)
    result = fit_be_q_equation_pointwise(
        u, q, 2e-4, spacings, spatial_derivative_method="spectral",
        time_discretization="two_point", rhs_time_level="previous",
        include_terms=names, threshold=0.0, optimizer_alpha=0.0,
    )
    np.testing.assert_allclose(result.coefficients, [expected[n] for n in names], rtol=2e-8, atol=2e-9)
    assert result.relative_residual < 1e-10


def test_2d_local_weak_form_coefficient_recovery_on_exact_discrete_trajectory():
    u, q, spacings, expected = _synthetic_exact_trajectory(nx=30, ny=32)
    names = tuple(expected)
    result = fit_be_q_equation_local_weak_form(
        u, q, 2e-4, spacings, spatial_derivative_method="spectral",
        time_discretization="two_point", rhs_time_level="previous",
        include_terms=names, patch_half_widths=(4, 4), patch_strides=(5, 5),
        threshold=0.0, optimizer_alpha=0.0,
    )
    np.testing.assert_allclose(result.coefficients, [expected[n] for n in names], rtol=2e-7, atol=2e-8)
    assert result.relative_residual < 1e-9


def _synthetic_velocity_trajectory(nt=5, nx=30, ny=32, dt=1e-4, lx=7.5, ly=9.1):
    x = np.arange(nx)[:, None] * lx / nx
    y = np.arange(ny)[None, :] * ly / ny
    qxx = 0.07*np.cos(2*np.pi*x/lx) + 0.029*np.sin(4*np.pi*y/ly)
    qxy = 0.051*np.sin(2*np.pi*x/lx - 2*np.pi*y/ly) + 0.018*np.cos(4*np.pi*x/lx)
    q = reconstruct_q_tensor(np.stack((qxx, qxy), axis=-1))
    kx1 = 2*np.pi/lx
    ky1 = 2*np.pi/ly
    ux = 0.13*ky1*np.sin(2*np.pi*x/lx)*np.cos(2*np.pi*y/ly) - 0.04*ky1*np.cos(4*np.pi*x/lx - 2*np.pi*y/ly)
    uy = -0.13*kx1*np.cos(2*np.pi*x/lx)*np.sin(2*np.pi*y/ly) - 0.08*kx1*np.cos(4*np.pi*x/lx - 2*np.pi*y/ly)
    u = np.stack((ux, uy), axis=-1)
    spacings = (lx/nx, ly/ny)
    coeff = {
        "linear_velocity": -0.23,
        "material": -0.61,
        "laplacian": 0.37,
        "q_divergence": -0.44,
    }
    us = [u.copy()]
    qs = [q.copy()]
    for _ in range(nt - 1):
        basis = {
            "linear_velocity": u,
            "material": velocity_material_derivative(
                u, spacings, spatial_axes=(0, 1), spatial_derivative_method="spectral"
            ),
            "laplacian": velocity_laplacian(
                u, spacings, spatial_axes=(0, 1), spatial_derivative_method="spectral"
            ),
            "q_divergence": active_q_divergence_force(
                q, spacings, spatial_axes=(0, 1), spatial_derivative_method="spectral"
            ),
        }
        rhs = sum(coeff[name] * basis[name] for name in coeff)
        u = u + dt * rhs
        us.append(u.copy())
        qs.append(q.copy())
    return np.stack(us), np.stack(qs), spacings, coeff


def test_2d_velocity_pointwise_coefficient_recovery_on_exact_discrete_trajectory():
    u, q, spacings, expected = _synthetic_velocity_trajectory()
    names = tuple(expected)
    result = fit_be_equation_velocity_pointwise(
        u, q, 1e-4, spacings,
        include_terms=names,
        spatial_derivative_method="spectral",
        time_discretization="two_point",
        rhs_time_level="previous",
        threshold=0.0,
        optimizer_alpha=0.0,
    )
    np.testing.assert_allclose(result.coefficients, [expected[n] for n in names], rtol=2e-8, atol=2e-9)
    assert result.relative_residual < 1e-10


def test_2d_velocity_local_weak_form_coefficient_recovery_on_exact_discrete_trajectory():
    u, q, spacings, expected = _synthetic_velocity_trajectory(nx=36, ny=38)
    names = tuple(expected)
    result = fit_be_equation_velocity_local_weak_form(
        u, q, 1e-4, spacings,
        include_terms=names,
        spatial_derivative_method="spectral",
        time_discretization="two_point",
        rhs_time_level="previous",
        patch_half_widths=(4, 4),
        patch_strides=(5, 5),
        threshold=0.0,
        optimizer_alpha=0.0,
    )
    np.testing.assert_allclose(result.coefficients, [expected[n] for n in names], rtol=3e-7, atol=3e-8)
    assert result.relative_residual < 1e-9
