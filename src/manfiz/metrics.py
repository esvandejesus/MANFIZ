"""Channelwise metrics in the physical output units."""
import numpy as np
from .data import matrix


def regression_metrics(y, center, radius=None, *, tolerance=1e-10):
    y, center = matrix(y, 'y'), matrix(center, 'center')
    if y.shape != center.shape:
        raise ValueError('y and center shapes must match')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    err = y-center
    total = ((y-y.mean(0))**2).sum(0)
    r2 = np.divide((err**2).sum(0), total, out=np.full(y.shape[1], np.nan), where=total > 0)
    out = dict(RMSE=np.sqrt(np.mean(err**2, axis=0)), MAE=np.mean(abs(err), axis=0), R2=1-r2)
    if radius is not None:
        r = np.broadcast_to(np.asarray(radius, dtype=float), y.shape)
        if np.any(r < 0) or not np.all(np.isfinite(r)):
            raise ValueError('Interval radii must be finite and nonnegative')
        excess = abs(err)-r
        width = 2*r.mean(0)
        span = np.ptp(y, axis=0)
        out.update(PICP=np.mean(excess <= tolerance, axis=0), MPIW=width,
                   NMPIW=np.divide(width, span, out=np.full_like(span, np.nan), where=span > 0),
                   Violations=np.sum(excess > tolerance, axis=0),
                   MaxExcess=np.maximum(excess.max(0), 0), MinSlack=(-excess).min(0))
    return out
