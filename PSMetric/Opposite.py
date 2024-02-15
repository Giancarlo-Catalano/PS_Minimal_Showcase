import numpy as np

from PSMetric.Metric import Metric
from PS import PS
from PRef import PRef


class Opposite(Metric):
    metric: Metric

    def __init__(self, metric: Metric):
        self.metric = metric
        super().__init__()

    def __repr__(self):
        return f"Opposite({self.metric})"

    def set_pRef(self, pRef: PRef):
        self.metric.set_pRef(pRef)

    def get_single_score(self, ps: PS) -> float:
        return -self.metric.get_single_score(ps)
