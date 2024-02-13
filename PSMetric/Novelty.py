import numpy as np

from PSMetric.Metric import Metric
from PS import PS, STAR
from PRef import PRef


class Novelty(Metric):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Novelty"

    def get_single_score(self, ps: PS, pRef: PRef) -> float:
        return np.log(1+float(len(pRef.fitnesses_of_observations(ps))))
