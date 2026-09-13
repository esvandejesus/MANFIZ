"""Generalized-bell grid premises and joint multi-output variable projection."""
from dataclasses import dataclass
from itertools import product
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, lstsq
from scipy.optimize import minimize
from scipy.special import logsumexp
from .data import matrix


def affine(x):
    return np.column_stack([x, np.ones(len(x))])


def design_matrix(phi, weights):
    return (weights[:, :, None]*phi[:, None, :]).reshape(len(phi), -1)


def ridge_fit(phi, weights, y, ridge):
    h = design_matrix(phi, weights)
    gram = h.T@h
    gram.flat[::len(gram)+1] += ridge
    rhs = h.T@y
    try:
        beta = cho_solve(cho_factor(gram, lower=True, check_finite=False), rhs,
                         check_finite=False)
        if not np.all(np.isfinite(beta)):
            raise np.linalg.LinAlgError('Nonfinite ridge coefficients')
    except np.linalg.LinAlgError:
        ha = np.vstack([h, np.sqrt(ridge)*np.eye(h.shape[1])])
        ya = np.vstack([y, np.zeros((h.shape[1], y.shape[1]))])
        beta = lstsq(ha, ya, lapack_driver='gelsy', check_finite=False)[0]
    return beta.reshape(weights.shape[1], phi.shape[1], y.shape[1])


def evaluate_center(phi, weights, beta):
    return np.einsum('ki,kj,ijo->ko', weights, phi, beta, optimize=True)


@dataclass
class BellPremises:
    counts: tuple
    parameters: np.ndarray  # one [a,b,c] row per membership, grouped by input
    antecedents: np.ndarray

    @classmethod
    def initialize(cls, x, counts):
        x = matrix(x)
        if np.isscalar(counts):
            counts = (counts,)*x.shape[1]
        if any(not np.isfinite(n) or int(n) != n for n in counts):
            raise ValueError('Membership counts must be integers')
        counts = tuple(int(n) for n in counts)
        if len(counts) != x.shape[1] or any(n < 2 for n in counts):
            raise ValueError('At least two memberships per premise variable are required')
        params = []
        for j, n in enumerate(counts):
            lo, hi = x[:, j].min(), x[:, j].max()
            if hi-lo <= 1e-12:
                raise ValueError('Constant premise input: remove it or choose other premise columns')
            width = (hi-lo)/(2*(n-1))
            params.extend([width, 2., c] for c in np.linspace(lo, hi, n))
        rules = np.array(list(product(*(range(n) for n in counts))), dtype=int)
        return cls(counts, np.asarray(params, dtype=float), rules)

    def pack(self):
        p = self.parameters.copy(); p[:, :2] = np.log(p[:, :2])
        return p.ravel()

    def unpack(self, p):
        p = np.asarray(p, dtype=float).reshape(-1, 3)
        if p.shape != self.parameters.shape or not np.all(np.isfinite(p)):
            raise ValueError('Invalid packed premise parameters')
        out = p.copy()
        out[:, 0] = np.exp(np.clip(p[:, 0], np.log(1e-3), np.log(50)))
        out[:, 1] = np.exp(np.clip(p[:, 1], np.log(.1), np.log(20)))
        return out

    def weights(self, x, parameters=None):
        x = matrix(x)
        if x.shape[1] != len(self.counts):
            raise ValueError('Incorrect number of premise variables')
        pars = self.parameters if parameters is None else parameters
        raw_log = np.zeros((len(x), len(self.antecedents)))
        offset = 0
        for j, count in enumerate(self.counts):
            p = pars[offset:offset+count]
            with np.errstate(divide='ignore'):
                logdistance = np.log(np.abs(x[:, j:j+1]-p[:, 2]))-np.log(p[:, 0])
            log_mu = -np.logaddexp(0., 2*p[:, 1]*logdistance)
            raw_log += log_mu[:, self.antecedents[:, j]]
            offset += count
        w = np.exp(raw_log-logsumexp(raw_log, axis=1, keepdims=True))
        if not np.all(np.isfinite(w)):
            raise FloatingPointError('Nonfinite normalized rule weights')
        return w


def train_premises(xp, x, y, counts, config):
    config.validate()
    premise = BellPremises.initialize(xp, counts)
    phi = affine(x); p0 = premise.pack(); history = []
    start = time.perf_counter()

    def objective(p):
        pars = premise.unpack(p)
        w = premise.weights(xp, pars)
        beta = ridge_fit(phi, w, y, config.ridge)
        err = y-evaluate_center(phi, w, beta)
        z = p.reshape(-1, 3)
        penalty = (np.maximum(abs(z[:, 0])-4, 0)**2).mean()
        penalty += (np.maximum(abs(z[:, 1])-3, 0)**2).mean()
        penalty += (np.maximum(abs(z[:, 2])-4, 0)**2).mean()
        data = float(np.mean(err*err)); value = data+config.penalty*penalty
        history.append((value, data, config.penalty*float(penalty)))
        return value

    initial = objective(p0)
    if config.optimize:
        opt = minimize(objective, p0, method='Nelder-Mead', options={
            'maxiter':config.max_iter, 'maxfev':config.max_evaluations,
            'xatol':config.x_tolerance, 'fatol':config.objective_tolerance,
            'adaptive':config.adaptive_simplex})
        pstar = opt.x
        termination = dict(success=bool(opt.success), status=int(opt.status),
                           message=str(opt.message), iterations=int(opt.nit),
                           function_evaluations=int(opt.nfev))
    else:
        pstar = p0
        termination = dict(success=False, status=-1, message='Premise optimization disabled',
                           iterations=0, function_evaluations=0)
    premise.parameters = premise.unpack(pstar)
    w = premise.weights(xp)
    beta = ridge_fit(phi, w, y, config.ridge)
    final = objective(pstar)
    report = dict(**termination, initial_objective=initial, final_objective=final,
                  final_data_objective=history[-1][1], final_penalty=history[-1][2],
                  n_rules=len(premise.antecedents), n_premise_parameters=len(p0),
                  initial_parameters=p0, final_parameters=pstar,
                  objective_history=np.asarray(history), seconds=time.perf_counter()-start,
                  initialization='Training range; evenly spaced centers, half-spacing widths, b=2',
                  optimizer='SciPy Nelder-Mead; joint normalized multi-output objective')
    return premise, beta, report
