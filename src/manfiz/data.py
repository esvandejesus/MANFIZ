"""Causal NARX construction and training-only normalization."""
from dataclasses import dataclass
import numpy as np


def matrix(value, name='array'):
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        value = value[:, None]
    if value.ndim != 2 or not value.size or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be a nonempty finite 2-D array')
    return value


def bound_vector(value, size, name):
    value = np.broadcast_to(np.asarray(value, dtype=float), (size,)).copy()
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    constant_columns: np.ndarray

    @classmethod
    def fit(cls, x):
        x = matrix(x)
        if len(x) < 2:
            raise ValueError('Sample standard deviation requires at least two rows')
        mean, scale = x.mean(0), x.std(0, ddof=1)
        constant = scale <= np.finfo(float).eps * np.maximum(1, np.abs(mean))
        scale[constant] = 1.0
        return cls(mean, scale, constant)

    def transform(self, x):
        x = matrix(x)
        if x.shape[1] != len(self.mean):
            raise ValueError('Feature/output dimension does not match fitted normalization')
        return (x-self.mean)/self.scale

    def inverse(self, x):
        return matrix(x)*self.scale+self.mean


@dataclass
class RegressionData:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sample_indices: np.ndarray


@dataclass(frozen=True)
class NARXSpec:
    output_lags: tuple = (1, 2)
    input_lags: tuple = (1, 2)
    discard: int = 30

    def __post_init__(self):
        for lags in (self.output_lags, self.input_lags):
            if not lags or any(int(v) != v or v <= 0 for v in lags):
                raise ValueError('Lags must be nonempty sequences of positive integers')
            if len(set(lags)) != len(lags):
                raise ValueError('Repeated lag entries are not allowed')
        if self.discard < 0 or int(self.discard) != self.discard:
            raise ValueError('discard must be a nonnegative integer')

    @property
    def start(self):
        return max(self.discard, max(self.output_lags), max(self.input_lags))

    def transform(self, u, y, *, group=0):
        u, y = matrix(u, 'u'), matrix(y, 'y')
        if len(u) != len(y) or len(y) <= self.start:
            raise ValueError('Aligned trajectories longer than the lag/discard horizon are required')
        k = np.arange(self.start, len(y))
        columns = [y[k-lag, o] for o in range(y.shape[1]) for lag in self.output_lags]
        columns += [u[k-lag, j] for j in range(u.shape[1]) for lag in self.input_lags]
        return RegressionData(np.column_stack(columns), y[k].copy(),
                              np.full(len(k), group), k)

    def transform_runs(self, runs):
        """Each (u,y) tuple is a distinct experiment; lags never cross runs."""
        parts = [self.transform(u, y, group=i) for i, (u, y) in enumerate(runs)]
        if not parts:
            raise ValueError('At least one experiment is required')
        return RegressionData(*(np.concatenate([getattr(p, name) for p in parts], axis=0)
                                for name in ('X', 'y', 'groups', 'sample_indices')))

    def input_premise_columns(self, n_inputs, n_outputs):
        offset = n_outputs*len(self.output_lags)
        return tuple(offset+j*len(self.input_lags) for j in range(n_inputs))

    def regressor_noise_bounds(self, n_inputs, sensor_bound):
        by = np.asarray(sensor_bound, dtype=float).ravel()
        if not by.size or np.any(by < 0) or not np.all(np.isfinite(by)):
            raise ValueError('Invalid sensor bounds')
        return np.r_[np.repeat(by, len(self.output_lags)),
                     np.zeros(n_inputs*len(self.input_lags))]

    def free_run(self, model, u, initial_y, *, start=None, nominal=False):
        """Predict centers recursively, using only initial_y[:start] as history."""
        u, initial_y = matrix(u, 'u'), matrix(initial_y, 'initial_y')
        start = self.start if start is None else int(start)
        if start < max(max(self.output_lags), max(self.input_lags)) or start >= len(u):
            raise ValueError('Invalid free-run starting index')
        if len(initial_y) < start:
            raise ValueError('Insufficient initial output history')
        ny = initial_y.shape[1]
        pred = np.empty((len(u), ny)); pred[:start] = initial_y[:start]
        for k in range(start, len(u)):
            x = [pred[k-lag, o] for o in range(ny) for lag in self.output_lags]
            x += [u[k-lag, j] for j in range(u.shape[1]) for lag in self.input_lags]
            pred[k] = model.predict(np.asarray(x)[None, :], nominal=nominal)[0]
        return pred[start:]
