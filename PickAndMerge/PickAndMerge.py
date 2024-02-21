import random

import numpy as np

from FullSolution import FullSolution
from PS import PS
from PSMiners.Individual import Individual, tournament_select
from SearchSpace import SearchSpace


class FSSampler:
    search_space: SearchSpace
    individuals: list[Individual]
    merge_limit: int

    tournament_size = 3

    def __init__(self,
                 search_space: SearchSpace,
                 individuals: list[Individual],
                 merge_limit: int):
        self.search_space = search_space
        self.basis_PSs = individuals
        self.merge_limit = merge_limit

    def sample_ps_unsafe(self) -> PS:
        """ this is unsafe in the sense that the result might not be complete"""

        available = list(self.individuals)

        def pick() -> Individual:
            return tournament_select(available, self.tournament_size)

        current = PS.empty(self.search_space)
        added_count = 0

        while (len(available) > 0) and (added_count < self.merge_limit) and not current.is_fully_fixed():
            to_add = pick()
            if PS.mergeable(current, to_add.ps):
                current = PS.merge(current, to_add.ps)
                added_count += 1
            available.remove(to_add)

        return current  # it might be incomplete!!

    def fill_in_the_gaps(self, incomplete_ps: PS):
        result_values = np.array(incomplete_ps.values)
        for index in incomplete_ps.get_unfixed_variable_positions():
            result_values[index] = random.randrange(self.search_space.cardinalities[index])
        return PS(result_values)

    def sample(self) -> FullSolution:
        produced_ps = self.sample_ps_unsafe()
        filled_feature = self.fill_in_the_gaps(produced_ps)
        return filled_feature.to_FS()
