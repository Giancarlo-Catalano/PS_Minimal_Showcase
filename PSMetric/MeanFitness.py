import warnings
from typing import Optional

import numpy as np

from PSMetric.Metric import Metric
from PS import PS
from PRef import PRef


class MeanFitness(Metric):
    pRef: Optional[PRef]


    def __init__(self):
        super().__init__()
        self.pRef = None


    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

    def __repr__(self):
        return "MeanFitness"

    def get_single_score(self, ps: PS) -> float:
        observed_fitnesses = self.pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            #warnings.warn(f"The passed PS {ps} has no observations, and thus the MeanFitness could not be calculated")
            return -1

        return np.average(observed_fitnesses)
