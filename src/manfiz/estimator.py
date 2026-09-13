"""Multi-output MANFIZ estimator, calibration and portable persistence."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import numpy as np
from .config import FitConfig
from .data import matrix, bound_vector, Standardizer
from .premises import BellPremises, train_premises, affine, evaluate_center
from .calibration import prediction_space_prior, joint_training_bounds, verify_witness
from .recursion import train_output
from .zonotopes import weight_matrix
from .io import jsonable, fingerprint


@dataclass
class IntervalPrediction:
    center: np.ndarray
    radius: np.ndarray
    components: dict

    @property
    def lower(self):
        return self.center-self.radius

    @property
    def upper(self):
        return self.center+self.radius


class MANFIZ:
    """Shared bell premises and independent rule-local zonotopes per output.

    All public data and bounds use physical units. ``premise_columns`` are
    zero-based columns of the supplied regressor matrix, not raw input indices.
    ``fit`` always reinitializes and trains the premises and all consequent sets.
    """
    def __init__(self, premise_columns, n_memberships=2, *, config=None, weight=None):
        cols = tuple(premise_columns)
        if not cols or any(int(c) != c or c < 0 for c in cols) or len(set(cols)) != len(cols):
            raise ValueError('premise_columns must be distinct nonnegative integer indices')
        self.premise_columns = tuple(int(c) for c in cols)
        self.n_memberships = n_memberships
        self.config = deepcopy(config) if config is not None else FitConfig()
        self.weight = None if weight is None else np.asarray(weight, dtype=float).copy()

    def _require_nominal(self):
        if not hasattr(self, 'beta_'):
            raise RuntimeError('Call fit or fit_nominal before prediction')

    def _require_fitted(self):
        self._require_nominal()
        if not hasattr(self, 'outputs_'):
            raise RuntimeError('Zonotopes are not fitted; call fit or fit_zonotopes')

    def features(self, X):
        """Return normalized affine regressors and normalized rule weights."""
        self._require_nominal()
        xn = self.x_scaler_.transform(X)
        return affine(xn), self.premises_.weights(xn[:, self.premise_columns])

    def fit_nominal(self, X, y):
        X, y = matrix(X, 'X'), matrix(y, 'y')
        if len(X) != len(y) or max(self.premise_columns) >= X.shape[1]:
            raise ValueError('Unaligned data or premise column outside X')
        self.config.validate(X.shape[1]+1)
        self.weight_ = weight_matrix(self.weight, X.shape[1]+1)
        for name in ('beta_', 'outputs_', 'prior_', 'feasibility_', 'validation_bound_', 'noise_calibration_'):
            self.__dict__.pop(name, None)
        self.x_scaler_, self.y_scaler_ = Standardizer.fit(X), Standardizer.fit(y)
        xn, yn = self.x_scaler_.transform(X), self.y_scaler_.transform(y)
        premises, beta, report = train_premises(xn[:, self.premise_columns], xn, yn,
                                               self.n_memberships, self.config.premise)
        self.premises_, self.beta_ = premises, beta
        self.n_features_in_, self.n_outputs_ = X.shape[1], y.shape[1]
        self.training_fingerprint_ = fingerprint(X, y)
        self.fit_report_ = dict(premise=report, training_rows=len(X),
                                training_fingerprint=self.training_fingerprint_,
                                trained_from_scratch=True)
        return self

    def fit_zonotopes(self, X, y, *, groups, sensor_bound):
        """Refit all sets using the same data as fit_nominal; useful for q studies."""
        self._require_nominal()
        X, y = matrix(X, 'X'), matrix(y, 'y')
        if fingerprint(X, y) != self.training_fingerprint_:
            raise ValueError('Consequent fitting must use the same X,y as premise training')
        self.config.validate(self.n_features_in_+1)
        self.sensor_bound_ = bound_vector(sensor_bound, self.n_outputs_, 'sensor_bound')
        phi, w = self.features(X); yn = self.y_scaler_.transform(y)
        started = time.perf_counter()
        self.prior_ = prediction_space_prior(phi, w, yn, groups, self.beta_,
                                             self.sensor_bound_/self.y_scaler_.scale, self.config)
        self.feasibility_ = joint_training_bounds(phi, w, yn, self.beta_, self.prior_['radii'],
                                                  self.y_scaler_.scale, self.sensor_bound_,
                                                  self.config.regression_bound_safety)
        self.regression_bound_ = np.array([a['bound_physical'] for a in self.feasibility_])
        outputs = []
        for o in range(self.n_outputs_):
            out = train_output(phi, w, yn[:, o], self.beta_[:, :, o],
                               self.prior_['radii'][:, :, o],
                               self.regression_bound_[o]/self.y_scaler_.scale[o],
                               self.config, self.weight_)
            if self.config.verify_training_witness and self.config.parameter_drift == 0:
                out['witness_membership_errors'] = verify_witness(
                    out['centers'], out['generators'], self.feasibility_[o]['witness'])
            outputs.append(out)
        self.outputs_ = outputs
        self.fit_report_['consequent_seconds'] = time.perf_counter()-started
        self.fit_report_['accepted_updates'] = [len(o['history']) for o in outputs]
        self.fit_report_['regression_bound'] = self.regression_bound_.copy()
        self.fit_report_['witness_checked'] = bool(self.config.verify_training_witness and
                                                  self.config.parameter_drift == 0)
        for name in ('validation_bound_', 'noise_calibration_'):
            self.__dict__.pop(name, None)
        return self

    def fit(self, X, y, *, groups, sensor_bound):
        self.fit_nominal(X, y)
        return self.fit_zonotopes(X, y, groups=groups, sensor_bound=sensor_bound)

    def coefficients(self, *, nominal=False):
        self._require_nominal()
        beta = self.beta_.copy()
        if not nominal:
            self._require_fitted()
            for o, out in enumerate(self.outputs_):
                beta[:, :, o] += np.asarray(out['centers'])
        return beta

    def predict(self, X, *, nominal=False):
        phi, w = self.features(X)
        return self.y_scaler_.inverse(evaluate_center(phi, w, self.coefficients(nominal=nominal)))

    def _parameter_interval(self, X, chunk_size):
        self._require_fitted()
        X = matrix(X, 'X')
        if int(chunk_size) != chunk_size or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        center = np.empty((len(X), self.n_outputs_)); radius = np.zeros_like(center)
        beta = self.coefficients()
        for start in range(0, len(X), chunk_size):
            sl = slice(start, start+chunk_size)
            phi, w = self.features(X[sl])
            center[sl] = self.y_scaler_.inverse(evaluate_center(phi, w, beta))
            for o, out in enumerate(self.outputs_):
                for i, r in enumerate(out['generators']):
                    radius[sl, o] += abs(w[:, i])*abs(phi@r).sum(1)*self.y_scaler_.scale[o]
        return center, radius

    def _history_radius(self, X):
        _, w = self.features(X)
        beta = self.coefficients()
        bx = self.noise_calibration_['regressor_bound']/self.x_scaler_.scale
        out = np.zeros((len(w), self.n_outputs_))
        for o, fitted in enumerate(self.outputs_):
            center_slopes = abs(w@beta[:, :-1, o])
            support = np.array([abs(r[:-1]).sum(1) for r in fitted['generators']])
            out[:, o] = ((center_slopes+abs(w)@support)@bx)*self.y_scaler_.scale[o]
        return out

    def predict_interval(self, X, *, mode='validation', chunk_size=2048):
        """Modes: parameters, validation, noise. Calibration is always explicit."""
        center, radius = self._parameter_interval(X, chunk_size)
        components = {'parameters': radius.copy()}
        if mode == 'validation':
            if not hasattr(self, 'validation_bound_'):
                raise RuntimeError('Call calibrate_validation before mode=validation')
            components['validation_defect'] = np.broadcast_to(self.validation_bound_, radius.shape).copy()
        elif mode == 'noise':
            if not hasattr(self, 'noise_calibration_'):
                raise RuntimeError('Call calibrate_noise before mode=noise')
            cal = self.noise_calibration_
            components['clean_defect'] = np.broadcast_to(cal['clean_defect'], radius.shape).copy()
            components['current_noise'] = np.broadcast_to(cal['sensor_bound'], radius.shape).copy()
            components['history_noise'] = self._history_radius(X)
        elif mode != 'parameters':
            raise ValueError('mode must be parameters, validation, or noise')
        total = np.sum(list(components.values()), axis=0)
        return IntervalPrediction(center, total, components)

    def calibrate_validation(self, X, y, *, safety=1.05):
        """Conservative physical residual box calibrated on a separate run."""
        if not np.isfinite(safety) or safety < 1:
            raise ValueError('safety must be finite and >= 1')
        y = matrix(y, 'y'); center = self.predict(X)
        if y.shape != center.shape:
            raise ValueError('Calibration target shape mismatch')
        self.validation_bound_ = safety*abs(y-center).max(0)
        self.__dict__.pop('noise_calibration_', None)
        self.fit_report_['validation_calibration'] = dict(rows=len(y), safety=safety,
                                                         fingerprint=fingerprint(X,y))
        return self

    def calibrate_noise(self, clean_X, clean_y, *, sensor_bound, regressor_bound,
                        safety=1.10, minimum_model_defect=None):
        """Freeze a clean envelope and bounded measurement/history-noise extension.

        The resulting guarantee is conditional on the clean envelope holding.
        Premise inputs must have zero uncertainty. Calibration alone is not a
        proof of that envelope on unseen trajectories; see docs/mathematics.md.
        """
        if not np.isfinite(safety) or safety < 1:
            raise ValueError('safety must be finite and >= 1')
        self._require_fitted()
        by = bound_vector(sensor_bound, self.n_outputs_, 'sensor_bound')
        bx = bound_vector(regressor_bound, self.n_features_in_, 'regressor_bound')
        if np.any(bx[list(self.premise_columns)] != 0):
            raise ValueError('Noise extension requires noise-free premise columns')
        y = matrix(clean_y, 'clean_y')
        pred = self.predict_interval(clean_X, mode='parameters')
        if pred.center.shape != y.shape:
            raise ValueError('Clean calibration target shape mismatch')
        deficit = np.maximum(abs(y-pred.center)-pred.radius, 0).max(0)
        if minimum_model_defect is None:
            floor = getattr(self, 'validation_bound_', np.zeros(self.n_outputs_))
        else:
            floor = bound_vector(minimum_model_defect, self.n_outputs_, 'minimum_model_defect')
        self.noise_calibration_ = dict(sensor_bound=by, regressor_bound=bx,
                                       clean_defect=np.maximum(safety*deficit, floor),
                                       maximum_calibration_deficit=deficit,
                                       minimum_model_defect=floor.copy(), safety=safety,
                                       calibration_rows=len(y), fingerprint=fingerprint(clean_X,y),
                                       guarantee='Conditional on clean enclosure and the declared deterministic noise bounds')
        return self

    def save(self, path):
        """Write a versioned NPZ model using JSON metadata and numeric arrays."""
        self._require_fitted()
        path = Path(path)
        if path.suffix.lower() != '.npz':
            raise ValueError('Model path must end in .npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(format='manfiz-npz', format_version=1, config=asdict(self.config),
                        premise_columns=self.premise_columns, counts=self.premises_.counts,
                        training_fingerprint=self.training_fingerprint_, fit_report=self.fit_report_,
                        prior=self.prior_, feasibility=self.feasibility_,
                        output_reports=[{k:v for k,v in o.items() if k not in ('centers','generators')}
                                        for o in self.outputs_],
                        noise_calibration=getattr(self, 'noise_calibration_', None))
        arrays = dict(beta=self.beta_, premise_parameters=self.premises_.parameters,
                      antecedents=self.premises_.antecedents, weight=self.weight_,
                      mean_x=self.x_scaler_.mean, scale_x=self.x_scaler_.scale,
                      constant_x=self.x_scaler_.constant_columns,
                      mean_y=self.y_scaler_.mean, scale_y=self.y_scaler_.scale,
                      constant_y=self.y_scaler_.constant_columns,
                      sensor_bound=self.sensor_bound_, regression_bound=self.regression_bound_)
        if hasattr(self, 'validation_bound_'):
            arrays['validation_bound'] = self.validation_bound_
        for o, fitted in enumerate(self.outputs_):
            arrays[f'centers_{o}'] = np.asarray(fitted['centers'])
            for i, r in enumerate(fitted['generators']):
                arrays[f'generators_{o}_{i}'] = r
        arrays['metadata'] = np.asarray(json.dumps(jsonable(metadata), allow_nan=False))
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path):
        """Read only this toolbox's NPZ format; never unpickle objects."""
        with np.load(path, allow_pickle=False) as file:
            meta = json.loads(str(file['metadata'].item()))
            if meta.get('format') != 'manfiz-npz' or meta.get('format_version') != 1:
                raise ValueError('Unsupported MANFIZ model format/version')
            model = cls(meta['premise_columns'], tuple(meta['counts']),
                        config=FitConfig.from_dict(meta['config']), weight=file['weight'])
            def numeric(name):
                a = file[name].copy()
                if not np.all(np.isfinite(a)):
                    raise ValueError(f'Nonfinite model array: {name}')
                return a
            model.beta_ = numeric('beta')
            if model.beta_.ndim != 3:
                raise ValueError('Invalid coefficient tensor')
            nrules, ntheta, ny = model.beta_.shape
            model.n_features_in_, model.n_outputs_ = ntheta-1, ny
            model.config.validate(ntheta)
            model.weight_ = weight_matrix(model.weight, ntheta)
            model.x_scaler_ = Standardizer(numeric('mean_x'), numeric('scale_x'), numeric('constant_x'))
            model.y_scaler_ = Standardizer(numeric('mean_y'), numeric('scale_y'), numeric('constant_y'))
            for scaler, size in ((model.x_scaler_,ntheta-1),(model.y_scaler_,ny)):
                if any(a.shape != (size,) for a in (scaler.mean,scaler.scale,scaler.constant_columns)) or np.any(scaler.scale <= 0):
                    raise ValueError('Invalid normalization dimensions or scales')
            counts = tuple(meta['counts']); pars = numeric('premise_parameters'); rules = numeric('antecedents')
            if (len(counts) != len(model.premise_columns) or max(model.premise_columns) >= ntheta-1
                or any(n < 2 or int(n) != n for n in counts) or np.prod(counts) != nrules
                or pars.shape != (sum(counts),3) or np.any(pars[:,:2] <= 0)
                or rules.shape != (nrules,len(counts)) or np.any(rules != rules.astype(int))
                or np.any(rules < 0) or np.any(rules >= np.asarray(counts))):
                raise ValueError('Invalid premise arrays')
            model.premises_ = BellPremises(counts, pars, rules.astype(int))
            model.sensor_bound_ = bound_vector(numeric('sensor_bound'),ny,'sensor_bound')
            model.regression_bound_ = bound_vector(numeric('regression_bound'),ny,'regression_bound')
            model.outputs_ = []
            for o in range(ny):
                out = meta['output_reports'][o]; centers = numeric(f'centers_{o}')
                if centers.shape != (nrules,ntheta):
                    raise ValueError('Invalid center dimensions')
                rs = [numeric(f'generators_{o}_{i}') for i in range(nrules)]
                if any(r.ndim != 2 or r.shape[0] != ntheta or r.shape[1] < ntheta for r in rs):
                    raise ValueError('Invalid generator dimensions')
                out.update(centers=list(centers), generators=rs); model.outputs_.append(out)
            if 'validation_bound' in file:
                model.validation_bound_ = bound_vector(numeric('validation_bound'),ny,'validation_bound')
            if meta['noise_calibration'] is not None:
                cal = meta['noise_calibration']
                for key, size in [('sensor_bound',ny),('regressor_bound',ntheta-1),('clean_defect',ny),
                                  ('maximum_calibration_deficit',ny),('minimum_model_defect',ny)]:
                    cal[key] = bound_vector(cal[key],size,key)
                if np.any(cal['regressor_bound'][list(model.premise_columns)] != 0):
                    raise ValueError('Invalid noise bounds on premise columns')
                model.noise_calibration_ = cal
            model.training_fingerprint_ = meta['training_fingerprint']
            model.fit_report_ = meta['fit_report']
            model.prior_, model.feasibility_ = meta['prior'], meta['feasibility']
            return model
