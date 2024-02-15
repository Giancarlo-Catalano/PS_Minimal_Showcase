from typing import Optional

import numpy as np

from PRef import PRef
from PS import PS
from PSMetric.Metric import Metric


class Novelty(Metric):
    pRef: Optional[PRef]

    def __init__(self):
        super().__init__()
        self.pRef = None

    def __repr__(self):
        return "Novelty"

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

    def get_single_score(self, ps: PS) -> float:
        return np.log(1 + float(len(self.pRef.fitnesses_of_observations(ps))))
