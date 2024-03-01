from typing import Optional

import numpy as np

from PRef import PRef
from PS import PS, STAR
from PSMetric.Metric import Metric
from SearchSpace import SearchSpace
from PSMetric.MeanFitness import MeanFitness


class Novelty(Metric):
    pRef: Optional[PRef]
    normalised_pRef: Optional[PRef]

    def __init__(self):
        super().__init__()
        self.pRef = None
        self.normalised_pRef = None

    def __repr__(self):
        return "Novelty"

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.normalised_pRef = self.pRef.get_with_normalised_fitnesses()

    def get_single_normalised_score(self, ps: PS) -> float:
        observations = self.normalised_pRef.fitnesses_of_observations(ps)
        if len(observations) == 0:
            return 1

        expected_observations = self.pRef.sample_size
        for value, cardinality in zip(ps.values, self.pRef.search_space.cardinalities):
            if value != STAR:
                expected_observations /= cardinality

        if len(observations) >= expected_observations:
            return 0

        return 1.0 - len(observations) / expected_observations  # DEBUG

        mean_fitness_bonus = 2 * np.average(observations)
        # the mean fitness bonus (beta) is a value between 0 and 2, since the mean fitness (mu) is in [0, 1]
        # mu < 0.5 ==> beta > 1, malus because the average fitness is below the average

        return 1 - ((len(observations) / expected_observations) ** (1 / mean_fitness_bonus))

    def get_single_score(self, ps: PS) -> float:
        return self.get_single_normalised_score(ps)


