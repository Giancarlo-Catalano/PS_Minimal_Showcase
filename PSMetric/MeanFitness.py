from typing import Optional

import numpy as np

import utils
from PRef import PRef
from PS import PS
from PSMetric.Metric import Metric


class MeanFitness(Metric):
    pRef: Optional[PRef]
    normalised_pRef: Optional[PRef]


    def __init__(self):
        super().__init__()
        self.pRef = None
        self.normalised_pRef = None

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

    def __repr__(self):
        return "MeanFitness"



    def get_single_score(self, ps: PS) -> float:
        observed_fitnesses = self.pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            #warnings.warn(f"The passed PS {ps} has no observations, and thus the MeanFitness could not be calculated")
            return -1

        return np.average(observed_fitnesses)


    def get_single_normalised_score(self, ps: PS) -> float:
        observed_fitnesses = self.normalised_pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            # warnings.warn(f"The passed PS {ps} has no observations, and thus the MeanFitness could not be calculated")
            return 0

        return np.average(observed_fitnesses)
