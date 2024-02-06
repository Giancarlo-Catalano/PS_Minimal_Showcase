import warnings

import numpy as np

from PSMetric.Metric import Metric
from PS import PS
from PRef import PRef


class MeanFitness(Metric):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "MeanFitness"

    def get_single_unnormalised_score(self, ps: PS, pRef: PRef) -> float:
        observed_fitnesses = pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            warnings.warn(f"The passed PS {ps} has no observations, and thus the MeanFitness could not be calculated")
            return -1

        return np.average(observed_fitnesses)
