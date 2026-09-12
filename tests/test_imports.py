def test_public_package_imports():
    import activebe_validator
    import activebe_validator.io
    import activebe_validator.operators
    import activebe_validator.terms
    import activebe_validator.three_d
    import activebe_validator.two_d
    import activebe_validator.validation

    assert activebe_validator.__version__ == "0.1.0"


def test_focused_public_namespaces_expose_expected_entry_points():
    from activebe_validator.io import discover_q_from_processed_npy
    from activebe_validator.terms.q import project_symmetric_traceless
    from activebe_validator.validation.q import fit_be_q_equation_pointwise

    assert callable(discover_q_from_processed_npy)
    assert callable(project_symmetric_traceless)
    assert callable(fit_be_q_equation_pointwise)


def test_three_d_namespace_groups_public_api():
    from activebe_validator import three_d

    assert three_d.io is not None
    assert three_d.terms is not None
    assert three_d.validation is not None
