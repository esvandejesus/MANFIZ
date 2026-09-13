# MANFIZ Toolbox

Multi-output neuro-fuzzy identification with zonotopic consequents, implemented
in Python. `MANFIZ.fit()` **trains the shared premises, nominal consequents, and
the zonotopes for each output from scratch**. It does not require MATLAB,
Octave, the Fuzzy Logic Toolbox, or `.mat` files.

This version includes three newly trained models, reproducible data,
comparisons, figures, and 21 automated tests. Its scope is system identification
and interval prediction; it does not include fault-detection schemes.

## Installation

Python 3.10 or later is required. From the project directory:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[all]"
```

The core package uses NumPy and SciPy. `[plots]` adds Matplotlib and
`[comparisons]` adds scikit-learn. The wheel included in `dist/` allows the core
package to be installed without the source tree; models, examples, and results
are included in the project ZIP archive.

## Training and Prediction

```python
import numpy as np
from manfiz import MANFIZ, NARXSpec
from manfiz.benchmarks import make_training_runs

spec = NARXSpec()
runs = make_training_runs(system=1)
train = spec.transform_runs([(r.u, r.y) for r in runs[:4]])
validation = runs[4].regression(spec)
test = runs[5].regression(spec)

model = MANFIZ(premise_columns=(4, 6, 8), n_memberships=2)
model.fit(train.X, train.y, groups=train.groups,
          sensor_bound=np.sqrt(3) * 0.01)
model.calibrate_validation(validation.X, validation.y)

interval = model.predict_interval(test.X, mode="validation")
center, lower, upper = interval.center, interval.lower, interval.upper
model.save("results/my_model.npz")
loaded = MANFIZ.load("results/my_model.npz")
```

`premise_columns` uses zero-based Python indices over **X**, the regressor
matrix. `sensor_bound` is a deterministic maximum amplitude expressed in
physical units. A standard deviation is not, by itself, a maximum bound.

For custom data, `examples/custom_data.py` accepts NPZ files containing `u`
with shape `(N, n_inputs)` and `y` with shape `(N, n_outputs)`, one file per
experiment. At least two training experiments with enough samples are required
to calibrate the prior shape. Experiment boundaries are preserved when lagged
regressors are constructed. The Cartesian product of membership functions grows
exponentially with the number of premise variables.

## Noise-Aware Intervals

Three explicit interval modes are available:

| Mode | Radius added around the MANFIZ center |
|---|---|
| `parameters` | Consequent-parameter uncertainty |
| `validation` | Parameter radius + validation residual box |
| `noise` | Parameter radius + clean-model defect + current noise + lagged-regressor noise propagation |

For `noise`, a clean-output envelope and deterministic bounds must first be
frozen:

```python
# clean_cal is RegressionData obtained from noise-free calibration trajectories.
# It must be kept separate from the confirmation data.
b = np.sqrt(3) * 0.02
bx = spec.regressor_noise_bounds(3, np.array([b, b]))
model.calibrate_noise(clean_cal.X, clean_cal.y,
                      sensor_bound=b, regressor_bound=bx)
interval = model.predict_interval(test.X, mode="noise")
```

This block illustrates the API; the complete experiment that generates
`clean_cal` while preserving data separation is implemented in
`manfiz-benchmark`. Premise variables must be noise-free for this extension.
For real-world data, a clean trajectory or a valid clean-output envelope
requires independent justification; the toolbox does not infer it from noisy
measurements.

**Coverage of all bounded-noise realizations is conditional on the validity of
the clean-output envelope.** A finite study with no violations does not prove
universal coverage for all future trajectories. Untruncated Gaussian noise has
unbounded support and therefore does not admit a finite deterministic bound that
contains all possible values.

## Reproducing the Included Results

The following command reproduces the computational budget used for the reported
study. Use a new output directory because the program protects existing
experiments.

```bash
# Linux/macOS: fix the thread count to reduce timing variability.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 manfiz-benchmark \
  --output results/my_run --max-iter 6000 --max-evaluations 10000

# Then run the following on the newly trained models:
python examples/budget_sensitivity.py --results results/my_run
python examples/quantile_comparison.py --results results/my_run
python tools/verify_saved_results.py --results results/my_run
python -m unittest discover -s tests -v
```

On Windows, set `$env:OPENBLAS_NUM_THREADS="1"` and
`$env:OMP_NUM_THREADS="1"`, then run `manfiz-benchmark` on a single line.
The default API budget is 2000/3500; the reported experiment explicitly used
6000/10000. Random seeds are stored in `protocol.json`.

## Results from the Current Release

| System | Iterations | Objective-function reduction | Termination |
|---|---:|---:|---|
| 1 | 6000 | 5.79% | Iteration limit |
| 2 | 6000 | 24.44% | Iteration limit |
| 3 | 4286 | 6.44% | Numerical convergence |

A total of 2406 local consequent updates were performed across the six outputs.
Joint feasibility and membership of the common witness in all 48 final
zonotopes were verified numerically. Across 540 independent confirmation
trajectories, there were **0 violations among 1,911,600 measured output
values**, using excitation amplitudes 1.00/1.15/1.30 and bounded uniform-noise
standard deviations 0.005/0.010/0.020. The mean interval width increased from
0.386877 to 0.508028 after adding the explicit noise-propagation extension, an
increase of 31.32%. These values correspond to prediction instants
`k=30,...,1799`. The first 30 samples are used as initial history and are
excluded from the reported prediction metrics.

The reduction procedure used the minimum-violation fallback in 653 of 712
events. The complexity cap and outer inclusion are preserved, but the prescribed
inflation/contraction thresholds are not guaranteed in those fallback events.
The result files record these events explicitly.

## Files and Documentation

| Path | Contents |
|---|---|
| `src/manfiz/` | Reusable API and algorithms |
| `examples/` | Training, custom data, budget studies, quantile comparison, and free-run simulation |
| `tests/` | Geometry, causality, training, noise, and equivalence checks |
| `results/fresh_training/` | Three newly trained models, data, seeds, traces, metrics, and figures |
| `results/budget_sensitivity/` | Nine consequent fits using newly trained shared premises |
| `results/quantile_comparison/` | Quantile-based comparison using the same features |
| `docs/mathematics.md` | Equations, variables, assumptions, and limitations |
| `docs/api.md` | API parameters and usage |
| `docs/matlab_mapping.md` | MATLAB/Python functional correspondence |
| `docs/reproducibility.md` | Protocol, versions, seeds, and commands |
| `docs/training_report.md` | Findings, corrections, and results from the new training runs |

The code and results are prepared for public repository distribution.
The publication license is defined by the author; see `NOTICE.md`.
`CITATION.cff` contains the manuscript authorship information without inventing
a DOI or repository URL.

## Summary

MANFIZ is a native Python research toolbox for shared-premise multi-output
neuro-fuzzy identification with zonotopic consequents. The full `fit` call
initializes and optimizes the premises, refits nominal affine coefficients,
calibrates a jointly feasible bounded regression-error model, and fits the local
zonotopes. It supports bounded-noise one-step interval prediction, model
serialization, and recursive center simulation. The supplied study is freshly
trained and is not a replay of MATLAB parameters. Coverage claims are
conditional on a valid clean-output envelope; see the equations and limitations
in `docs/mathematics.md`.
