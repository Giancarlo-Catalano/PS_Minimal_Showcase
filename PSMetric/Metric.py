from typing import Iterable

import numpy as np

from PRef import PRef
from PS import PS
from custom_types import ArrayOfFloats


class Metric:
    def __repr__(self):
        """ Return a string which describes the Criterion, eg 'Robustness' """
        raise Exception("Error: a realisation of PSMetric does not implement __repr__")


    def set_pRef(self, pRef: PRef):
        raise Exception("Error: a realisation of PSMetric does not implement set_pRef")

    def get_single_score(self, ps: PS) -> float:
        raise Exception("Error: a realisation of PSMetric does not implement get_single_score_for_PS")

    def get_single_normalised_score(self, ps: PS) -> float:#
        raise Exception("Error: a realisation of PSMetric does not implement get_single_normalised_score")

    def get_unnormalised_scores(self, pss: Iterable[PS], pRef: PRef) -> ArrayOfFloats:
        """default implementation, subclasses might overwrite this"""
        return np.array([self.get_single_score(ps, pRef) for ps in pss])



