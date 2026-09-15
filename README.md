# ActiveBE Validator

The package provides separate pure-2D and 3D validation APIs. The 2D
implementation has additionally been cross-validated against an independent
PSANC2D GPU solver to double-precision roundoff. See
[`docs/validation_2d.md`](docs/validation_2d.md) and
[`docs/validation_3d.md`](docs/validation_3d.md) for validation details.

`activebe-validator` is a lightweight Python package for validating active
Beris-Edwards simulation outputs by reconstructing equation terms, recovering
coefficients, and evaluating residuals independently of the simulation code.

## Current scope

- Q-tensor equation checks
- velocity-equation checks with active forcing and passive backflow
- pointwise/strong-form fits
- projected and local weak-form fits
- processed-output discovery workflows
- cross-solver validation using ANS, Ludwig, and similar active-nematics solvers

Both pure 2D and 3D implementations are available. Dimension-independent
differential operators are shared internally, while the user-facing API is
dimension-explicit.

## Package layout

```text
src/activebe_validator/
    two_d/              # pure-2D Q, velocity, I/O, and validation API
    three_d/            # public 3D API
    operators/          # shared differential operators
    q_terms.py          # internal 3D Q-term implementation
    velocity_terms.py   # internal 3D velocity/stress implementation
    q_checker.py        # internal 3D Q regression implementation
    ns_checker.py       # internal 3D velocity regression implementation
    discovery.py        # processed-output workflow implementation
```

## Installation

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

## Public API

The package root is deliberately small. User code should choose the spatial
dimension explicitly:

```python
from activebe_validator import three_d, two_d

result_3d = three_d.fit_be_q_equation_pointwise(...)
result_2d = two_d.fit_be_q_equation_pointwise(...)
```

The 3D namespace also exposes the main velocity-equation fits and processed-Q
discovery workflow:

```python
from activebe_validator import three_d

three_d.fit_be_equation_velocity_pointwise(...)
three_d.fit_be_equation_velocity_local_weak_form(...)
three_d.discover_q_from_processed_npy(...)
```

Pure 2D support is available under `activebe_validator.two_d`. Here "2D" means
both a two-dimensional spatial grid and a symmetric-traceless `2 x 2` Q tensor:

```python
from activebe_validator import two_d

q = two_d.reconstruct_q_tensor(q_components)  # (..., Qxx, Qxy) -> (..., 2, 2)
result = two_d.fit_be_q_equation_pointwise(...)
```

The pure-2D Landau-de Gennes model uses `a2`, `a4`, and `kappa`. Because
`Tr(Q^3) = 0` identically for a traceless `2 x 2` tensor, there is no `a3`
cubic invariant or corresponding quadratic molecular-field term.

Top-level implementation modules such as `q_checker`, `ns_checker`, `q_terms`,
and `velocity_terms` are internal implementation details rather than supported
public API. New code should use `two_d` or `three_d`.

## Tutorials

The full worked tutorials live in `examples/tutorials/`:

- `q_validator_3d_introduction.ipynb` explains the 3D Q-equation validator,
  coefficient recovery, the processed-`.npy` workflow, and ANS/Ludwig examples.
- `velocity_validator_3d_introduction.ipynb` explains the 3D velocity validator,
  local weak forms, pressure elimination, viscosity normalization, passive
  backflow, and a Ludwig example.

Pure-2D usage is summarized in `examples/tutorials/README.md` and validated in
`docs/validation_2d.md`. The 2D API is intentionally separate because a
traceless symmetric `2 x 2` Q tensor has no cubic invariant `Tr(Q^3)`.

## Input convention

The high-level NumPy workflow expects chronological 3D stacks:

```text
Q components: (T, Nx, Ny, Nz, 5)  -> (Qxx, Qxy, Qxz, Qyy, Qyz)
velocity:     (T, Nx, Ny, Nz, 3)  -> (ux, uy, uz)
```

The full symmetric traceless Q tensor is reconstructed internally.

## Development checks

```bash
pytest
ruff check src tests
black --check src tests
python -m build
```
