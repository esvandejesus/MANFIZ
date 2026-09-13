"""Zonotope geometry and weighted, feasibility-aware outer reduction."""
from dataclasses import dataclass
import time
import numpy as np


def weight_matrix(weight, dimension):
    w = np.eye(dimension) if weight is None else np.asarray(weight, dtype=float)
    if w.shape != (dimension, dimension) or not np.all(np.isfinite(w)):
        raise ValueError('Weight matrix has incorrect dimensions or nonfinite values')
    if not np.allclose(w, w.T, rtol=0, atol=1e-12):
        raise ValueError('Weight matrix must be symmetric')
    np.linalg.cholesky(w)
    return w


def weighted_radius(generators, weight=None):
    r = np.asarray(generators, dtype=float)
    w = weight_matrix(weight, r.shape[0])
    return float(np.sqrt(max(0., np.sum((w@r)*r))))


@dataclass
class Zonotope:
    center: np.ndarray
    generators: np.ndarray

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float).reshape(-1)
        self.generators = np.asarray(self.generators, dtype=float)
        if (self.generators.ndim != 2 or self.generators.shape[0] != len(self.center)
                or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.generators))):
            raise ValueError('Invalid zonotope dimensions or values')

    @property
    def interval(self):
        radius = np.abs(self.generators).sum(1)
        return self.center-radius, self.center+radius

    def linear_map(self, a, offset=None):
        a = np.atleast_2d(np.asarray(a, dtype=float))
        center = a@self.center
        if offset is not None:
            center = center+np.asarray(offset, dtype=float)
        return Zonotope(center, a@self.generators)

    def minkowski_sum(self, other):
        if len(self.center) != len(other.center):
            raise ValueError('Minkowski summands must have equal dimension')
        return Zonotope(self.center+other.center,
                        np.column_stack([self.generators, other.generators]))

    def reduce(self, budget, weight=None):
        return Zonotope(self.center.copy(), reduce_fixed(self.generators, budget, weight))


def reduce_fixed(r, budget, weight=None):
    r = np.asarray(r, dtype=float)
    n, columns = r.shape
    if int(budget) != budget or budget < n:
        raise ValueError('Generator budget must be an integer >= state dimension')
    w = weight_matrix(weight, n)
    if columns <= budget:
        return r.copy()
    order = np.argsort(-np.sum((w@r)*r, axis=0), kind='stable')
    keep = int(budget)-n
    rs = r[:, order]
    return np.column_stack([rs[:, :keep], np.diag(np.abs(rs[:, keep:]).sum(1))])


def gain_and_covariation(p, c, error_bound, *, allow_invalid=False):
    """Joseph-form generator covariation; P is not a stochastic covariance."""
    p = (p+np.swapaxes(p, -2, -1))/2
    f = np.asarray(error_bound, dtype=float)
    l = p@c.T
    s = c@l+np.diag(f*f)
    s = (s+np.swapaxes(s, -2, -1))/2
    valid = np.linalg.cond(s, p=1) <= 1e14
    if np.ndim(p) == 2:
        if not valid:
            raise FloatingPointError('Innovation reciprocal condition number below 1e-14')
        k = np.linalg.solve(s, l.T).T
    else:
        if not np.all(valid) and not allow_invalid:
            raise FloatingPointError('An innovation matrix is ill-conditioned')
        k = np.zeros_like(l)
        if np.any(valid):
            k[valid] = np.swapaxes(np.linalg.solve(s[valid],
                np.swapaxes(l[valid], -2, -1)), -2, -1)
    a = np.eye(p.shape[-1])-k@c
    post = a@p@np.swapaxes(a, -2, -1)+(k*f)@np.swapaxes(k*f, -2, -1)
    return k, post, valid


def reduce_adaptive(r, c, error_bound, config, weight=None):
    n, columns = r.shape
    trigger = config.validate(n); w = weight_matrix(weight, n)
    p = r@r.T
    rho = np.sqrt(max(0., np.trace(w@p)))
    ref = max(rho, np.finfo(float).eps)
    info = dict(triggered=False, selected_budget=columns, fallback=False,
                candidates=0, search_seconds=0., inflation=1., selection_score=0.)
    if columns <= trigger:
        return r, p, info
    start = time.perf_counter()
    order = np.argsort(-np.sum((w@r)*r, axis=0), kind='stable')
    rs = r[:, order]
    prefix_abs = np.column_stack([np.zeros(n), np.cumsum(abs(rs), axis=1)])
    outer = np.einsum('ik,jk->kij', rs, rs)
    prefix_p = np.concatenate([np.zeros((1,n,n)), np.cumsum(outer, axis=0)])
    budgets = np.arange(config.q_target, min(config.q_max, columns-1)+1)
    keep = budgets-n
    tail = np.maximum(0., prefix_abs[:, -1]-prefix_abs[:, keep].T)
    candidates = prefix_p[keep].copy()
    candidates[:, np.arange(n), np.arange(n)] += tail*tail
    _, post, valid = gain_and_covariation(candidates, c, error_bound, allow_invalid=True)
    if not np.any(valid):
        raise FloatingPointError('All reduction candidates have ill-conditioned innovations')
    infl = np.sqrt(np.maximum(0., np.einsum('ij,bji->b', w, candidates)))/ref
    cycle = np.sqrt(np.maximum(0., np.einsum('ij,bji->b', w, post)))/ref
    cycle[~valid] = np.inf
    feasible = ((infl <= config.max_inflation*(1+config.inflation_tolerance)) &
                (cycle <= config.max_cycle_ratio*(1+config.cycle_tolerance)))
    violation = (np.maximum(infl/config.max_inflation-1, 0)**2+
                 np.maximum(cycle/config.max_cycle_ratio-1, 0)**2)
    if np.any(feasible):
        selected = np.flatnonzero(feasible)[0]
    else:
        minimum = violation.min()
        selected = np.flatnonzero(violation <= minimum+
                                  config.tie_tolerance*max(1., abs(minimum)))[0]
    k = keep[selected]
    reduced = np.column_stack([rs[:, :k], np.diag(abs(rs[:, k:]).sum(1))])
    if reduced.shape[1] > config.q_max or reduced.shape[1] >= columns:
        raise AssertionError('Hard reduction complexity invariant failed')
    info.update(triggered=True, selected_budget=int(budgets[selected]),
                fallback=not bool(np.any(feasible)), candidates=len(budgets),
                search_seconds=time.perf_counter()-start,
                inflation=weighted_radius(reduced, w)/ref,
                predicted_cycle_ratio=float(cycle[selected]),
                selection_score=float(violation[selected]))
    return reduced, reduced@reduced.T, info


def measurement_update(center, generators, c, z, error_bound):
    p = generators@generators.T
    gain, joseph, _ = gain_and_covariation(p, c, error_bound)
    a = np.eye(len(center))-gain@c
    new_center = center+gain@(z-c@center)
    new_r = np.column_stack([a@generators, -gain*error_bound])
    covariation = new_r@new_r.T
    error = np.linalg.norm(covariation-joseph)/max(1., np.linalg.norm(covariation))
    if not np.all(np.isfinite(new_r)) or not np.all(np.isfinite(new_center)) or error > 1e-9:
        raise FloatingPointError('Zonotopic update or independent Joseph identity failed')
    return new_center, new_r, dict(joseph_error=float(error), gain_norm=float(np.linalg.norm(gain)))
