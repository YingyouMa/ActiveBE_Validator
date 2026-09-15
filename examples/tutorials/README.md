# Validator tutorials

The notebooks in this directory are the main worked introductions to
`activebe-validator`.

## 3D

- `q_validator_3d_introduction.ipynb` starts from processed Q and velocity
  stacks, explains the candidate Beris-Edwards Q-equation terms, runs the
  high-level 3D discovery helper, and discusses recovered coefficients and
  residual metrics.
- `velocity_validator_3d_introduction.ipynb` explains why the velocity equation
  is treated with a local divergence-free weak form, how pressure is removed,
  how viscosity normalization is interpreted, and how Q-equation parameters
  are reused to construct passive backflow.

Both notebooks use the public namespace

```python
from activebe_validator import three_d
```

and can use the small ANS and Ludwig fixtures under `examples/data/`.

## Pure 2D

Pure 2D means both a two-dimensional spatial grid and a traceless symmetric
`2 x 2` Q tensor. Use

```python
from activebe_validator import two_d

q = two_d.reconstruct_q_tensor(q_components)
q_fit = two_d.fit_be_q_equation_pointwise(...)
v_fit = two_d.fit_be_equation_velocity_local_weak_form(...)
```

The 2D candidate library is not obtained by mechanically reusing the 3D bulk
free energy: `Tr(Q^3) = 0` identically for a traceless `2 x 2` tensor, so there
is no `a3` cubic invariant or corresponding quadratic molecular-field term.
See `docs/validation_2d.md` for the numerical and independent-solver validation
of the 2D implementation.
