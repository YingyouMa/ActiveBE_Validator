def test_public_package_imports():
    import activebe_validator
    import activebe_validator.three_d
    import activebe_validator.two_d

    assert activebe_validator.__version__ == "0.1.0"


def test_focused_public_namespaces_expose_expected_entry_points():
    from activebe_validator import three_d, two_d

    assert callable(three_d.discover_q_from_processed_npy)
    assert callable(three_d.project_symmetric_traceless)
    assert callable(three_d.fit_be_q_equation_pointwise)
    assert callable(two_d.reconstruct_q_tensor)
    assert callable(two_d.fit_be_q_equation_pointwise)


def test_three_d_namespace_groups_public_api():
    from activebe_validator import three_d

    assert callable(three_d.discover_from_processed_npy_local_weak_form)
    assert callable(three_d.fit_be_equation_velocity_pointwise)
    assert callable(three_d.fit_be_q_equation_local_weak_form)
