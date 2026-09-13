"""MANFIZ: shared-premise neuro-fuzzy zonotopic system identification."""
from .config import FitConfig, PremiseConfig, ReductionConfig
from .data import NARXSpec, RegressionData, Standardizer
from .estimator import MANFIZ, IntervalPrediction
from .zonotopes import Zonotope
from .metrics import regression_metrics

__version__ = '1.0.0'
__all__ = ['MANFIZ', 'IntervalPrediction', 'FitConfig', 'PremiseConfig',
           'ReductionConfig', 'NARXSpec', 'RegressionData', 'Standardizer',
           'Zonotope', 'regression_metrics']
