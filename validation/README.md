# Scientific validation

This directory is for reproducible, repository-maintainer validation of
`activebe-validator` against representative simulation data.

The validation workflow is deliberately separate from the package's automated
test suite. Large Ludwig and ANS datasets are internal validation inputs rather
than tutorial fixtures or package dependencies. They are not required to
install or use `activebe-validator`.

Validation scripts should record enough information to reproduce each result:
the input dataset and solver, physical parameters, spatial and temporal
sampling, validator options, recovered coefficients, and quantitative fit
metrics. Final user-facing results should be collected in PDF validation guides
under `docs/`, with compact code excerpts, tables, and figures explaining how
to use the public API and how its output compares with known solver parameters.

The automated `tests/` suite remains the regression gate for code changes. It
should use only small synthetic or compact numerical fixtures and must not
depend on the large internal validation datasets. The PDFs complement these
tests by documenting validation on realistic simulation data; they do not
replace pytest.

## Intended workflow

1. Run the validation scripts locally against the internal Ludwig and/or ANS
   data.
2. Save machine-readable numerical results and the figures/tables needed for
   the report.
3. Run the ordinary pytest suite to catch implementation regressions.
4. Regenerate the corresponding PDF in `docs/` when scientific validation
   results or the documented public API change.

Do not commit multi-gigabyte simulation arrays merely to make the user-facing
documentation executable. The published package should remain lightweight.
