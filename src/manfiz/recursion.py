"""Generic rule-local propagate-reduce-update training for one output."""
import time
import numpy as np
from scipy.linalg import qr
from .zonotopes import reduce_adaptive, measurement_update, weight_matrix, weighted_radius


def train_output(phi, weights, y, beta, prior_radius, error_bound, config, weight=None):
    n, ntheta = phi.shape; nrules = weights.shape[1]
    config.validate(ntheta); weight = weight_matrix(weight, ntheta)
    if error_bound <= 0 or not np.isfinite(error_bound):
        raise ValueError('Normalized total regression bound must be positive')
    nominal = np.einsum('ki,kj,ij->k', weights, phi, beta, optimize=True)
    residual = y-nominal
    centers = [np.zeros(ntheta) for _ in range(nrules)]
    generators = [np.diag(prior_radius[i]) for i in range(nrules)]
    buffers = [[] for _ in range(nrules)]
    logs = []; assigned = np.zeros(nrules, dtype=int); rejected = np.zeros(nrules, dtype=int)
    drift = config.parameter_drift*np.eye(ntheta) if config.parameter_drift else np.empty((ntheta,0))
    start = time.perf_counter(); peak_bytes = sum(r.nbytes for r in generators)
    for row in range(n):
        own = int(np.argmax(weights[row]))
        if weights[row, own] < config.dominant_weight_min:
            continue
        assigned[own] += 1; buffers[own].append(row)
        if len(buffers[own]) < config.batch_size:
            continue
        tick = time.perf_counter()
        cand = np.asarray(buffers[own], dtype=int)
        c_cand = weights[cand, own:own+1]*phi[cand]
        if len(cand) <= config.batch_size:
            idx = cand
        else:
            piv = qr(c_cand.T, mode='economic', pivoting=True)[2]
            idx = cand[piv[:config.batch_size]]
        c = weights[idx, own:own+1]*phi[idx]
        singular = np.linalg.svd(c, compute_uv=False)
        pe = singular[-1]/max(singular[0], np.finfo(float).eps)
        rank = int(np.sum(singular > singular[0]*max(c.shape)*np.finfo(float).eps))
        if rank < ntheta or pe < config.pe_ratio_min:
            rejected[own] += 1
            if len(buffers[own]) > config.max_buffer_size:
                buffers[own].pop(0)
            continue
        z = residual[idx].copy()
        f = np.full(len(idx), error_bound)
        for other in range(nrules):
            if other != own:
                z -= weights[idx, other]*(phi[idx]@centers[other])
                f += abs(weights[idx, other])*abs(phi[idx]@generators[other]).sum(1)
        propagated = np.column_stack([generators[own], drift])
        base_radius = max(weighted_radius(propagated, weight), np.finfo(float).eps)
        reduced, _, info = reduce_adaptive(propagated, c, f, config.reduction, weight)
        center, r, diag = measurement_update(centers[own], reduced, c, z, f)
        cycle = weighted_radius(r, weight)/base_radius
        if r.shape[1] > config.reduction.q_max+len(idx):
            raise AssertionError('Posterior generator cap failed')
        if info['triggered'] and not info['fallback']:
            if cycle > config.reduction.max_cycle_ratio*(1+config.reduction.cycle_tolerance)+1e-12:
                raise AssertionError('Actual cycle exceeds feasible selected bound')
        centers[own], generators[own] = center, r
        used = set(idx.tolist()); buffers[own] = [i for i in buffers[own] if i not in used]
        peak_bytes = max(peak_bytes, sum(rr.nbytes for rr in generators))
        logs.append(dict(rule=own, sample=row, sample_indices=idx.tolist(), rank=rank,
                         pe_ratio=float(pe), prior_columns=propagated.shape[1],
                         posterior_columns=r.shape[1], cycle_ratio=float(cycle),
                         seconds=time.perf_counter()-tick, **diag, **info))
    if not logs:
        raise RuntimeError('No informative batches accepted; inspect excitation, premise weights, and PE settings')
    return dict(centers=centers, generators=generators, history=logs,
                assigned_samples=assigned, rejected_batches=rejected,
                seconds=time.perf_counter()-start, peak_generator_bytes=peak_bytes)
