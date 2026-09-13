"""Prediction-space priors and joint training-feasibility programs."""
import numpy as np
from scipy.optimize import linprog
from .premises import ridge_fit, evaluate_center, design_matrix


def prediction_space_prior(phi, weights, y_normalized, groups, beta, sensor_normalized, config):
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != len(phi):
        raise ValueError('groups must contain one experiment identifier per row')
    refits = []
    for group in np.unique(groups):
        take = groups == group
        if take.sum() >= config.min_samples_per_group:
            refits.append(ridge_fit(phi[take], weights[take], y_normalized[take],
                                    config.premise.ridge))
    if len(refits) < 2:
        raise ValueError('Prior calibration requires at least two sufficiently large experiment groups')
    spread = np.max(abs(np.stack(refits)-beta), axis=0)
    largest = spread.max(axis=1, keepdims=True)
    shape = np.maximum(spread, config.prior_shape_floor*largest)
    norm = np.linalg.norm(shape, axis=1, keepdims=True)
    shape = np.divide(shape, norm, out=np.full_like(shape, 1/np.sqrt(phi.shape[1])),
                      where=norm > np.finfo(float).eps)
    residual = y_normalized-evaluate_center(phi, weights, beta)
    denominator = np.einsum('ki,kj,ijo->ko', abs(weights), abs(phi), shape, optimize=True)
    need = np.maximum(abs(residual)-sensor_normalized, 0.)
    if np.any((denominator <= 100*np.finfo(float).eps) & (need > 1e-12)):
        raise ValueError('The prior shape cannot enclose a nonzero training residual')
    ratio = np.divide(need, denominator, out=np.zeros_like(need), where=denominator > 0)
    alpha = ratio.max(0)
    radii = np.maximum(shape*(config.prior_safety*alpha), config.prior_radius_floor)
    radius = np.einsum('ki,kj,ijo->ko', abs(weights), abs(phi), radii, optimize=True)
    if np.any(abs(residual) > radius+sensor_normalized+1e-12):
        raise AssertionError('Initial pointwise training enclosure failed')
    return dict(radii=radii, shape=shape, raw_spread=spread, alpha_raw=alpha,
                alpha_safe=config.prior_safety*alpha, groups_used=len(refits),
                pointwise_training_coverage=np.mean(abs(residual) <= radius+sensor_normalized+1e-12, axis=0))


def minimax_correction(h, residual, prior_radius=None):
    """Minimize max |residual-H delta| for one constant correction tuple."""
    n, p = h.shape
    a = np.vstack([np.column_stack([h, -np.ones(n)]),
                   np.column_stack([-h, -np.ones(n)])])
    b = np.r_[residual, -residual]
    if prior_radius is None:
        bounds = [(None,None)]*p+[(0,None)]
    else:
        radius = np.asarray(prior_radius).ravel()
        if len(radius) != p or np.any(radius < 0):
            raise ValueError('Invalid prior radius for the joint feasibility program')
        bounds = list(zip(-radius, radius))+[(0,None)]
    objective = np.r_[np.zeros(p), 1.]
    result = linprog(objective, A_ub=a, b_ub=b, bounds=bounds, method='highs',
                     options={'primal_feasibility_tolerance':1e-9,
                              'dual_feasibility_tolerance':1e-9})
    if not result.success:
        raise RuntimeError(f'Joint training feasibility failed: {result.message}')
    primal = max(0., float(np.max(a@result.x-b)))
    dual_value = float(b@result.ineqlin.marginals)
    for j, (lo, hi) in enumerate(bounds):
        if lo is not None:
            dual_value += lo*result.lower.marginals[j]
        if hi is not None:
            dual_value += hi*result.upper.marginals[j]
    gap = abs(float(result.fun)-dual_value)
    if primal > 1e-7 or gap > 1e-7:
        raise FloatingPointError('LP solution failed primal/dual numerical verification')
    return dict(minimum_error=float(result.x[-1]), witness=result.x[:-1],
                primal_violation=primal, dual_gap=gap,
                inequality_duals=result.ineqlin.marginals,
                lower_duals=result.lower.marginals, upper_duals=result.upper.marginals)


def joint_training_bounds(phi, weights, y, beta, prior, output_scale, sensor_bound, safety):
    h = design_matrix(phi, weights)
    e = y-evaluate_center(phi, weights, beta)
    audits = []
    for o in range(y.shape[1]):
        result = minimax_correction(h, e[:, o], prior[:, :, o])
        physical = result['minimum_error']*output_scale[o]
        result['minimum_error_physical'] = float(physical)
        result['bound_physical'] = float(max(sensor_bound[o], safety*physical,
                                              1e-10*output_scale[o]))
        result['sensor_only_training_feasible'] = bool(physical <= sensor_bound[o]+1e-9*output_scale[o])
        audits.append(result)
    return audits


def verify_witness(centers, generators, witness, tolerance=1e-8):
    witness = np.asarray(witness).reshape(len(centers), -1)
    errors = []
    for center, r, point in zip(centers, generators, witness):
        result = linprog(np.zeros(r.shape[1]), A_eq=r, b_eq=point-center,
                         bounds=[(-1,1)]*r.shape[1], method='highs',
                         options={'primal_feasibility_tolerance':1e-9,
                                  'dual_feasibility_tolerance':1e-9})
        if not result.success:
            raise AssertionError('A common training witness left a final local zonotope')
        error = float(np.max(abs(r@result.x-(point-center))))
        if error > tolerance:
            raise AssertionError('Witness membership residual exceeds tolerance')
        errors.append(error)
    return errors
