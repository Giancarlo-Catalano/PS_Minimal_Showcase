import numpy as np

from PSMetric.Metric import Metric
from PS import PS, STAR
from PRef import PRef


class Opposite(Metric):
    metric: Metric
    def __init__(self, metric: Metric):
        self.metric = metric
        super().__init__()

    def __repr__(self):
        return f"Opposite({self.metric})"

    def get_unnormalised_scores(self, pss: list[PS], pRef: PRef) -> float:
        return -self.metric.get_unnormalised_scores(pss, pRef)
