from jmetal.core.problem import IntegerProblem

from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity


class AtomicityEvaluator:
    normalised_pRef: PRef
    global_isolated_benefits: list[list[float]]



    def __init__(self, pRef: PRef):
        self.normalised_pRef = Atomicity.get_normalised_pRef(pRef)
        self.global_isolated_benefits = Atomicity.get_global_isolated_benefits(self.normalised_pRef)

    def evaluate_single(self, ps: PS) -> float:
        return Atomicity.get_single_score_knowing_information(ps, self.normalised_pRef, self.global_isolated_benefits)




