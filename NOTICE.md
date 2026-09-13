# MANFIZ Toolbox — Notice

## Project and provenance

MANFIZ is the native Python reference implementation of the research software
for multi-output adaptive neuro-fuzzy inference with zonotopic consequents and
deterministic set-based uncertainty estimation.

This repository contains the Python implementation, mathematical corrections,
verification utilities, benchmark examples, and reproducibility artifacts used
for the MANFIZ study. Historical MATLAB material and numerical fixtures may be
used only for cross-validation and regression testing; the current Python
toolbox performs native end-to-end training and does not require MATLAB,
Octave, the Fuzzy Logic Toolbox, or `.mat` model files.

## Copyright and license

Copyright (c) 2026 Esvan-Jesús Pérez-Pérez.

The MANFIZ source code is distributed under the BSD 3-Clause License. See the
`LICENSE` file in the repository root for the full license text.

## Third-party software

MANFIZ depends on third-party Python packages including NumPy and SciPy, with
optional functionality provided by Matplotlib and scikit-learn. These
dependencies are distributed separately and remain subject to their respective
licenses.

## Reproducibility fixtures

Files under `tests/fixtures/` are provided as numerical regression-test
references associated with the MANFIZ development and verification workflow.
They are not required for training new MANFIZ models.

## Repository and citation

Source repository:
https://github.com/esvandejesus/MANFIZ

Citation metadata are provided in `CITATION.cff`. A software DOI should be
added to `CITATION.cff` after the archived v1.0.0 release is deposited and a
DOI is issued (for example, through Zenodo).
