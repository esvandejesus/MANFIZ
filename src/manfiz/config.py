"""Explicit, serializable settings for shared-premise MANFIZ training."""
from dataclasses import dataclass, field
import math
from numbers import Integral


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


@dataclass
class PremiseConfig:
    max_iter: int = 2000
    max_evaluations: int = 3500
    x_tolerance: float = 1e-4
    objective_tolerance: float = 1e-6
    ridge: float = 1e-8
    penalty: float = 1e-4
    optimize: bool = True
    adaptive_simplex: bool = False

    def validate(self):
        _positive_integer(self.max_iter, 'max_iter')
        _positive_integer(self.max_evaluations, 'max_evaluations')
        if self.max_iter < 1 or self.max_evaluations < 1:
            raise ValueError("Optimizer budgets must be positive")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.ridge, self.x_tolerance, self.objective_tolerance)):
            raise ValueError("Ridge and optimizer tolerances must be finite and positive")
        if not math.isfinite(self.penalty) or self.penalty < 0:
            raise ValueError("Premise penalty must be finite and nonnegative")


@dataclass
class ReductionConfig:
    q_target: int = 100
    q_max: int = 1000
    q_trigger: int | None = None
    max_inflation: float = 1.10
    max_cycle_ratio: float = 1.005
    inflation_tolerance: float = 1e-10
    cycle_tolerance: float = 1e-9
    tie_tolerance: float = 1e-12

    def validate(self, n_parameters):
        trigger = self.q_max if self.q_trigger is None else self.q_trigger
        for name, value in (('q_target',self.q_target),('q_max',self.q_max),('q_trigger',trigger)):
            _positive_integer(value,name)
        if not n_parameters <= self.q_target <= trigger <= self.q_max:
            raise ValueError("Require n_parameters <= q_target <= q_trigger <= q_max")
        if self.max_inflation <= 1 or self.max_cycle_ratio <= 0:
            raise ValueError("Require max_inflation > 1 and max_cycle_ratio > 0")
        values = (self.max_inflation, self.max_cycle_ratio, self.inflation_tolerance,
                  self.cycle_tolerance, self.tie_tolerance)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Invalid reduction tolerances")
        return trigger


@dataclass
class FitConfig:
    premise: PremiseConfig = field(default_factory=PremiseConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    batch_size: int = 16
    max_buffer_size: int = 64
    pe_ratio_min: float = 1e-3
    dominant_weight_min: float = 0.20
    prior_shape_floor: float = 0.02
    prior_radius_floor: float = 1e-7
    prior_safety: float = 1.05
    min_samples_per_group: int = 100
    regression_bound_safety: float = 1.05
    parameter_drift: float = 0.0
    verify_training_witness: bool = True

    def validate(self, n_parameters):
        for name in ('batch_size','max_buffer_size','min_samples_per_group'):
            _positive_integer(getattr(self,name),name)
        self.premise.validate()
        self.reduction.validate(n_parameters)
        if not n_parameters <= self.batch_size <= self.max_buffer_size:
            raise ValueError("Require n_parameters <= batch_size <= max_buffer_size")
        if not 0 <= self.dominant_weight_min <= 1 or not 0 <= self.pe_ratio_min <= 1:
            raise ValueError("Weight and singular-value ratio thresholds must be in [0,1]")
        if self.min_samples_per_group < 2:
            raise ValueError("At least two samples per refit group are required")
        if self.prior_safety < 1 or self.regression_bound_safety < 1:
            raise ValueError("Safety factors must be >= 1")
        values = (self.prior_shape_floor, self.prior_radius_floor, self.parameter_drift,
                  self.prior_safety, self.regression_bound_safety)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Invalid prior or drift settings")
        if self.prior_radius_floor <= 0:
            raise ValueError("prior_radius_floor must be positive")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value['premise'] = PremiseConfig(**value.get('premise', {}))
        value['reduction'] = ReductionConfig(**value.get('reduction', {}))
        return cls(**value)
