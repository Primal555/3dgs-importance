"""Fixed-weight RD scoring; references are scales, never budget constraints."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FixedRateDistortion:
    beta: float
    d_ref: float
    c_ref: float

    def __post_init__(self):
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        if any(not math.isfinite(v) or v <= 0 for v in (self.d_ref, self.c_ref)):
            raise ValueError('RD references must be finite and positive')

    def score(self, distortion, cost):
        if any(not math.isfinite(v) or v < 0 for v in (distortion, cost)):
            raise ValueError('RD distortion/cost must be finite and nonnegative')
        return distortion/self.d_ref + self.beta*(cost/self.c_ref)

    def candidate_key(self, row):
        distortion, cost = row['source_mse'], row['total_channel_uses']
        return self.score(distortion, cost), distortion, cost
