# ActiveBE_Validator

Dedicated tooling for validating active Beris-Edwards equations against simulation outputs.

## Scope

This repository is intended to become a focused validator for:

- Q-tensor equation discovery and coefficient recovery
- velocity-equation discovery with active forcing and passive backflow
- strong-form and weak-form comparisons
- cross-solver consistency checks across ANS, Ludwig, and other active-nematics solvers

## Initial Direction

The first development phase will focus on:

1. defining a clean input format for processed simulation snapshots
2. separating strong-form and weak-form validators into explicit modules
3. building a reproducible benchmark suite with known-good ANS and Ludwig cases
4. adding reports that summarize recovered coefficients, residuals, and fit quality

## Status

Project scaffold created on August 7, 2026.
