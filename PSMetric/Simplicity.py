import numpy as np

from PSMetric.Metric import Metric
from PS import PS, STAR
from PRef import PRef


class Simplicity(Metric):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Simplicity"

    def get_single_score(self, ps: PS, pRef: PRef) -> float:
        return float(np.sum(ps.values == STAR))
