from PRef import PRef
from PSMetric.Atomicity import Atomicity


class AtomicityEvaluator:
    normalised_pRef: PRef
    global_isolated_benefits: list[list[float]]

    atomicity_object: Atomicity


    def __init__(self, pRef: PRef):
        self.normalised_pRef = Atomicity.get_normalised_pRef(pRef)
        self.global_isolated_benefits = Atomicity.get_global_isolated_benefits(self.normalised_pRef)
        self.atomicity_object = Atomicity()


