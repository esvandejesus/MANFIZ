"""Optional quantile baseline on exactly the same frozen fuzzy feature matrix."""
import numpy as np
from .data import matrix
from .premises import design_matrix
from .estimator import IntervalPrediction


class SharedFeatureQuantile:
    """Calibrated quantile baseline, not a deterministic enclosure theorem.

    Calibration uses an order statistic at ceil((n+1)*coverage). Classical
    conformal guarantees require exchangeability, which is not asserted for
    autocorrelated trajectories in these benchmarks.
    """
    def __init__(self, model, *, coverage=.95, alpha=1e-6):
        if not 0 < coverage < 1 or not np.isfinite(alpha) or alpha < 0:
            raise ValueError('Invalid coverage or regularization')
        self.model=model; self.coverage=coverage; self.alpha=alpha

    def _design(self,X):
        phi,w=self.model.features(X)
        return design_matrix(phi,w)

    def fit(self,X,y):
        from sklearn.linear_model import QuantileRegressor
        h=self._design(X); yn=self.model.y_scaler_.transform(y)
        if len(h) != len(yn):
            raise ValueError('Unaligned quantile data')
        tails=((1-self.coverage)/2,(1+self.coverage)/2)
        self.coef_=np.empty((2,h.shape[1],yn.shape[1]))
        for q,tau in enumerate(tails):
            for o in range(yn.shape[1]):
                fit=QuantileRegressor(quantile=tau,alpha=self.alpha,fit_intercept=False,
                                      solver='highs').fit(h,yn[:,o])
                self.coef_[q,:,o]=fit.coef_
        self.__dict__.pop('calibration_',None)
        return self

    def _raw(self,X):
        if not hasattr(self,'coef_'):
            raise RuntimeError('Fit quantile coefficients first')
        h=self._design(X)
        both=np.stack([self.model.y_scaler_.inverse(h@b) for b in self.coef_])
        return both.min(0),both.max(0)

    def calibrate(self,X,y):
        lo,hi=self._raw(X); y=matrix(y,'y')
        if y.shape != lo.shape:
            raise ValueError('Quantile calibration target shape mismatch')
        rank=int(np.ceil((len(y)+1)*self.coverage))
        if rank > len(y):
            raise ValueError('Insufficient calibration samples for requested coverage')
        scores=np.maximum(lo-y,y-hi)
        self.calibration_=np.maximum(np.sort(scores,axis=0)[rank-1],0)
        self.calibration_rank_=rank
        return self

    def predict_interval(self,X):
        if not hasattr(self,'calibration_'):
            raise RuntimeError('Calibrate the quantile interval before prediction')
        lo,hi=self._raw(X)
        center=(lo+hi)/2; raw=(hi-lo)/2
        return IntervalPrediction(center,raw+self.calibration_,
                                  {'raw_quantile':raw,'calibration':np.broadcast_to(self.calibration_,raw.shape)})
