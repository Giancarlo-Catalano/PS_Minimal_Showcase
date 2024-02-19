from typing import Optional

import numpy as np

import utils
from PRef import PRef
from PS import PS
from PSMetric.Metric import Metric


class MeanFitness(Metric):
    pRef: Optional[PRef]
    normalised_pRef: Optional[PRef]

    max_fitness: Optional[float]
    min_fitness: Optional[float]
    median_fitness: Optional[float]


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

        self.max_fitness = np.max(pRef.fitness_array)
        self.min_fitness = np.min(pRef.fitness_array)

    def __repr__(self):
        return "MeanFitness"



    def get_single_score(self, ps: PS) -> float:
        observed_fitnesses = self.pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            #warnings.warn(f"The passed PS {ps} has no observations, and thus the MeanFitness could not be calculated")
            return -1

        return np.average(observed_fitnesses)


    def get_single_normalised_score_old(self, ps: PS) -> float:
        observed_fitnesses = self.normalised_pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            # warnings.warn(f"The passed PS {ps} has no observations, and thus the MeanFitness could not be calculated")
            return 0

        return np.average(observed_fitnesses)

    def get_single_normalised_score(self, ps: PS) -> float:
        average_fitness = self.get_single_score(ps)
        return (average_fitness - self.min_fitness)/self.max_fitness



class ChanceOfGood(Metric):
    pRef: Optional[PRef]
    median_fitness: Optional[float]


    def __init__(self):
        super().__init__()
        self.pRef = None
        self.median_fitness = None
    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

        self.median_fitness = np.median(pRef.fitness_array)

    def __repr__(self):
        return "ChanceOfGood"

    def get_single_normalised_score(self, ps: PS) -> float:
        observations = self.pRef.fitnesses_of_observations(ps)
        if len(observations) == 0:
            return 0

        amount_which_are_better_than_median = sum([1 for observation in observations
                                                   if observation > self.median_fitness])

        return amount_which_are_better_than_median / len(observations)