# ActiveBE Validator

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

The current numerical implementation is the existing 3D validator. Dimension-independent finite-difference operators live in `activebe_validator.operators` so that a dedicated 2D implementation can be added without duplicating common numerical infrastructure.

## Package layout

```text
src/activebe_validator/
    operators/          # dimension-independent differential operators
    q_terms.py          # current 3D Beris-Edwards Q terms
    velocity_terms.py   # current 3D velocity/stress terms
    q_checker.py        # current 3D Q validators
    ns_checker.py       # current 3D velocity validators
    discovery.py        # processed-output workflows
    residuals.py
    two_d/              # reserved namespace for the next development phase
    three_d/            # explicit namespace re-exporting the current 3D API
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

The package root is deliberately small. Prefer focused imports:

```python
from activebe_validator.io import discover_q_from_processed_npy
from activebe_validator.terms.q import q_material_derivative
from activebe_validator.validation.q import fit_be_q_equation_pointwise
from activebe_validator.validation.velocity import fit_be_equation_velocity_pointwise
```

The current implementation is 3D and is also grouped explicitly as:

```python
from activebe_validator import three_d

three_d.terms
three_d.validation
three_d.io
```

Legacy implementation modules such as `q_checker`, `ns_checker`, `q_terms`, and
`velocity_terms` remain importable, but new user code should use the focused
namespaces above.

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
