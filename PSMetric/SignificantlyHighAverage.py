from typing import Optional

import numpy as np

import utils
from PRef import PRef
from PS import PS
from PSMetric.Metric import Metric
from scipy.stats import t


class SignificantlyHighAverage(Metric):
    pRef: Optional[PRef]
    normalised_pRef: Optional[PRef]
    normalised_population_mean: float

    def __init__(self):
        super().__init__()
        self.pRef = None
        self.normalised_pRef = None
        self.max_fitness = None
        self.min_fitness = None

    @staticmethod
    def get_normalised_pRef(pRef: PRef):
        normalised_fitnesses = utils.remap_array_in_zero_one(pRef.fitness_array)
        return PRef(full_solutions=pRef.full_solutions,
                    fitness_array=normalised_fitnesses,  # this is the only thing that changes
                    full_solution_matrix=pRef.full_solution_matrix,
                    search_space=pRef.search_space)

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.normalised_pRef = self.get_normalised_pRef(self.pRef)
        self.normalised_population_mean = np.average(self.normalised_pRef.fitness_array)

    def __repr__(self):
        return "SIA"

    def get_single_normalised_score(self, ps: PS) -> float:
        observations = self.normalised_pRef.fitnesses_of_observations(ps)
        n = len(observations)
        sample_mean = np.average(observations)
        sample_stdev = np.std(observations)

        if n < 1 or sample_stdev == 0:
            return 0

        t_score = (sample_mean - self.normalised_population_mean) / (sample_stdev / np.sqrt(n))
        cumulative_score = t.cdf(abs(t_score), df=n-1)  # p_value = 1 - cumulative_score

        def invert_and_augment(score: float):
            return 1. - np.sqrt(score * (2. - score))
            # return 1. - np.sqrt(1-np.square(score))

        return invert_and_augment(cumulative_score)
