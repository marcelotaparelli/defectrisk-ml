"""Deterministic abstention policy, independent of the probability model."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DefectProbability:
    probability: float
    model_name: str
    model_version: str
    calibration_method: str

    def __post_init__(self):
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError('A finite probability in [0,1] is required.')


@dataclass(frozen=True)
class PolicyDecision:
    outcome: str
    recommended_action: str
    policy_version: str


@dataclass(frozen=True)
class ConfidencePolicy:
    low_threshold: float | None
    high_threshold: float | None
    version: str = 'defectrisk-confidence-study-v1'

    def __post_init__(self):
        for threshold in [self.low_threshold, self.high_threshold]:
            if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
                raise ValueError('Thresholds must be None or finite in [0,1].')
        if self.low_threshold is not None and self.high_threshold is not None and self.low_threshold >= self.high_threshold:
            raise ValueError('LOW threshold must be strictly below HIGH threshold.')

    def decide(self, prediction: DefectProbability) -> PolicyDecision:
        p = prediction.probability
        if self.low_threshold is not None and p <= self.low_threshold:
            return PolicyDecision('LOW', 'Continue standard checks; lower estimated risk is not proof of clean code.', self.version)
        if self.high_threshold is not None and p >= self.high_threshold:
            return PolicyDecision('HIGH', 'Prioritize defect-focused inspection and testing.', self.version)
        return PolicyDecision('UNCERTAIN', 'Request human review before automated classification.', self.version)
