# Changelog

## 1.0.0

First public release of the native Python MANFIZ toolbox.

- Native end-to-end Python implementation of the MANFIZ workflow.
- Fresh training of shared premises and nominal consequents for all benchmark systems.
- Output-specific zonotopic consequent fitting and local PRU updates.
- Joint common-parameter feasibility calibration with primal/dual verification.
- Numerical verification of the common witness in all final rule-wise zonotopes.
- Explicit propagation of current measurement noise and lagged-regressor noise.
- Weighted outer zonotope reduction with feasibility-aware order selection and explicit fallback reporting.
- Causal NARX construction with experiment-boundary preservation.
- Training-only normalization and numerically stable rule-weight computation.
- Versioned NPZ/JSON model persistence; retraining invalidates prior calibrations.
- Validation of dimensions, deterministic bounds, and count-valued configuration parameters.
- Automated tests for geometry, causality, training, noise propagation, serialization, and MATLAB numerical-reference equivalence.
- Reproducibility scripts for generator-budget sensitivity, quantile comparison, and saved-result verification.
- Independent confirmation study covering 540 noisy trajectories and 1,911,600 measured output values with zero violations in the reported noise-aware setting.
- PNG/SVG figures, mathematical documentation, API documentation, and explicit reporting of optimization-budget termination for Systems 1 and 2.

## 0.1.0

Internal development version used while converting and auditing the original MATLAB workflow.
