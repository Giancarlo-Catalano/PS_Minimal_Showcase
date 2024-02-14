from typing import Optional

import numpy as np

from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity


class MetricEvaluator:

    def __init__(self):
        pass

    def evaluate_single(self, ps: PS) -> float:
        raise Exception("MetricEvaluator.evaluate_single(ps) was not implemented in a subclass")

    def setup(self, pRef: PRef):
        raise Exception("MetricEvaluator.setup(pRef) was not implemented in a subclass")


    def __repr__(self):
        raise Exception("MetricEvaluator.__repr__() was not implemented in a subclass")


class AtomicityEvaluator(MetricEvaluator):
    normalised_pRef: PRef
    global_isolated_benefits: list[list[float]]

    def __init__(self):
        super().__init__()

    def setup(self, pRef: PRef):
        self.normalised_pRef = Atomicity.get_normalised_pRef(pRef)
        self.global_isolated_benefits = Atomicity.get_global_isolated_benefits(self.normalised_pRef)

    def evaluate_single(self, ps: PS) -> float:
        return Atomicity.get_single_score_knowing_information(ps, self.normalised_pRef, self.global_isolated_benefits)

    def __repr__(self):
        return "Atomicity"


class SimplicityEvaluator(MetricEvaluator):
    def __init__(self):
        super().__init__()


    def setup(self, pRef: PRef):
        pass

    def evaluate_single(self, ps: PS):
        return len(ps) - ps.fixed_count()

    def __repr__(self):
        return "Simplicity"


class MeanFitnessEvaluator(MetricEvaluator):
    pRef: Optional[PRef]
    def __init__(self):
        super().__init__()
        self.pRef = None


    def setup(self, pRef: PRef):
        self.pRef = pRef

    def evaluate_single(self, ps: PS):
        observations = self.pRef.fitnesses_of_observations(ps)
        if len(observations) > 0:
            return np.average(observations)
        else:
            return -1

    def __repr__(self):
        return "MeanFitness"

