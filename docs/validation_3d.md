# Validation of the 3D implementation

The 3D implementation is tested at the operator and equation-recovery levels
using deterministic periodic fields. These tests are independent of any one
production simulation trajectory and are designed to detect sign, tensor-index,
time-staging, and regression-pipeline errors.

## Independent spectral operator checks

`tests/test_three_d_numerics.py` contains analytic Fourier-mode checks for the
3D differential operators. In particular, the spectral Laplacian is compared
against the exact `-k^2` eigenvalue of a single Fourier mode, and `div(Q)` is
compared against an independently assembled Fourier-space expression

```text
i k_j Q_ij(k).
```

These comparisons do not reuse the Validator implementation of the derivative
being checked.

## Q-equation coefficient recovery

A smooth symmetric-traceless 3D Q field and velocity field are evolved with an
exact discrete Euler update using a known linear combination of

```text
u.grad(Q)
Omega Q - Q Omega
ST(E Q + Q E)
Q
Q^2 - I Tr(Q^2)/3
Tr(Q^2) Q
laplacian(Q)
```

The known coefficients are then recovered from the saved trajectory using both
the pointwise and local weak-form APIs. The regression tests require residuals
near floating-point precision.

## Velocity-equation coefficient recovery

The 3D velocity tests construct a discrete trajectory with known coefficients
for

```text
u
(u.grad)u
laplacian(u)
div(Q).
```

Both the pointwise strong-form fit and the pressure-free local weak-form fit
recover the prescribed coefficients. The local weak-form test uses compactly
supported divergence-free test fields and finite-difference spatial operators,
matching the supported production code path.

## Regression value

Adding the 3D local weak-form test exposed a real implementation error in which
an accumulator name was overwritten by a NumPy field before `.append()` was
called. The bug was fixed and is now protected by the regression suite.

Unlike the pure-2D validation, the current 3D suite does not bundle an external
solver-level GPU fixture. The analytic operator checks and exact discrete
coefficient-recovery tests are therefore the portable CI validation layer; an
external 3D solver comparison can be maintained separately when a compact,
redistributable reference fixture is available.
