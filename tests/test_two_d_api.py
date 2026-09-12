import inspect


def test_two_d_public_api_imports():
    from activebe_validator import two_d

    assert callable(two_d.reconstruct_q_tensor)
    assert callable(two_d.fit_be_q_equation_pointwise)
    assert callable(two_d.fit_be_q_equation_local_weak_form)
    assert callable(two_d.fit_be_equation_velocity_pointwise)
    assert callable(two_d.fit_be_equation_velocity_local_weak_form)


def test_two_d_bulk_model_has_no_a3_parameter():
    from activebe_validator.two_d.q_terms import (
        landau_de_gennes_free_energy_density,
        q_molecular_field,
    )

    assert "a3" not in inspect.signature(landau_de_gennes_free_energy_density).parameters
    assert "a3" not in inspect.signature(q_molecular_field).parameters


def test_two_d_validation_library_has_no_bulk_quadratic_feature():
    from activebe_validator.two_d import validation

    assert "bulk_quadratic" not in validation._Q_FEATURE_NAMES
    assert "bulk_quartic" in validation._Q_FEATURE_NAMES
