from typing import Iterable

import numpy as np

from PRef import PRef
from PS import PS
from custom_types import ArrayOfFloats


class Metric:
    def __repr__(self):
        """ Return a string which describes the Criterion, eg 'Robustness' """
        raise Exception(f"Error: a realisation of PSMetric({self.__repr__()}) does not implement __repr__")

    def set_pRef(self, pRef: PRef):
        raise Exception(f"Error: a realisation of PSMetric({self.__repr__()}) does not implement set_pRef")

    def get_single_score(self, ps: PS) -> float:
        raise Exception(f"Error: a realisation of PSMetric({self.__repr__()}) does not implement get_single_score_for_PS")

    def get_single_normalised_score(self, ps: PS) -> float:  #
        raise Exception(f"Error: a realisation of PSMetric({self.__repr__()}) does not implement get_single_normalised_score")

    def get_unnormalised_scores(self, pss: Iterable[PS]) -> ArrayOfFloats:
        """default implementation, subclasses might overwrite this"""
        return np.array([self.get_single_score(ps) for ps in pss])


class ManyMetrics:
    metrics: list[Metric]
    used_evaluations: int

    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics
        self.used_evaluations = 0

    def get_labels(self) -> list[str]:
        return [m.__repr__() for m in self.metrics]

    def set_pRef(self, pRef: PRef):
        for m in self.metrics:
            m.set_pRef(pRef)

    def __repr__(self):
        return f"{self.get_labels()}"

    def get_scores(self, ps: PS) -> list[float]:
        self.used_evaluations += 1
        return [m.get_single_score(ps) for m in self.metrics]


    def get_amount_of_metrics(self) -> int:
        return len(self.metrics)
