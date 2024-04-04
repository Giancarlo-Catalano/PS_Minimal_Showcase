from typing import Iterable

import numpy as np

from PRef import PRef
from PS import PS
from custom_types import ArrayOfFloats


class Metric:
    used_evaluations: int

    def __init__(self):
        self.used_evaluations = 0
    def __repr__(self):
        """ Return a string which describes the Criterion, eg 'Robustness' """
        raise Exception(f"Error: a realisation of PSMetric does not implement __repr__")

    def set_pRef(self, pRef: PRef):
        raise Exception(f"Error: a realisation of PSMetric({self.__repr__()}) does not implement set_pRef")

    def get_single_score(self, ps: PS) -> float:
        raise Exception(f"Error: a realisation of PSMetric({self.__repr__()}) does not implement get_single_score_for_PS")

    def get_single_normalised_score(self, ps: PS) -> float:  #
        raise Exception(f"Error: a realisation of PSMetric({self.__repr__()}) does not implement get_single_normalised_score")

    def get_unnormalised_scores(self, pss: Iterable[PS]) -> ArrayOfFloats:
        """default implementation, subclasses might overwrite this"""
        return np.array([self.get_single_score(ps) for ps in pss])