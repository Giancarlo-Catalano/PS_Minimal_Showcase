from typing import Iterable

import numpy as np

from PRef import PRef
from PS import PS
from FullSolution import FullSolution
from SearchSpace import SearchSpace
from custom_types import ArrayOfFloats



class Metric:
    def __repr__(self):
        """ Return a string which describes the Criterion, eg 'Robustness' """
        raise Exception("Error: a realisation of PSMetric does not implement __repr__")

    def get_single_unnormalised_score(self, PS: PS, pRef: PRef) -> float:
        raise Exception("Error: a realisation of PSMetric does not implement get_single_score_for_PS")

    def get_unnormalised_scores(self, pss: Iterable[PS], pRef: PRef) -> ArrayOfFloats:
        """default implementation, subclasses might overwrite this"""
        return np.array([self.get_single_unnormalised_score(ps, pRef) for ps in pss])

    def get_normalised_scores(self, pss: Iterable[PS], pRef: PRef) -> ArrayOfFloats:
        """ Returns the scores which correlate with the criterion
            And they will all be in the range [0, 1]"""
        scores = self.get_unnormalised_scores(pss, pRef)
        return remap_array_in_zero_one(scores)
