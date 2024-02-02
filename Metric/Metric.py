import numpy as np

from PRef import PRef
from PS import PS
from FullSolution import FullSolution
from SearchSpace import SearchSpace
from custom_types import ArrayOfFloats

def remap_array_in_zero_one(input_array: np.ndarray):
    """remaps the values in the given array to be between 0 and 1"""
    min_value = np.min(input_array)
    max_value = np.max(input_array)

    if (min_value == max_value):
        return np.full(len(input_array), 0.5, dtype=float)  # all 0.5!

    return (input_array - min_value) / (max_value - min_value)

class Metric:
    def __repr__(self):
        """ Return a string which describes the Criterion, eg 'Robustness' """
        raise Exception("Error: a realisation of MeasurableCriterion does not implement __repr__")

    def get_unnormalised_score_for_PS(self, PS: PS, pRef: PRef) -> float:
        pass

    def get_normalised_scores_for_PSs(self, pss: Iterable[PS], pRef: PRef) -> ArrayOfFloats:
        """ Returns the scores which correlate with the criterion
            And they will all be in the range [0, 1]"""
        scores = [self.get_unnormalised_score_for_PS(ps, pRef) for ps in pss]
        return remap_array_in_zero_one(scores)