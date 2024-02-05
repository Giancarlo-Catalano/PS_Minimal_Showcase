from typing import Iterable

import numpy as np

import SearchSpace
import utils
from PSMetric.Metric import Metric
from PS import PS, STAR
from PRef import PRef
from custom_types import ArrayOfFloats


class Atomicity(Metric):
    def __init__(self):
        super().__init__()


    def __repr__(self):
        return "Atomicity"



    def get_isolated_in_search_space(self, search_space: SearchSpace.SearchSpace) -> list[PS]:
        empty: PS = PS.empty(search_space)
        return empty.specialisations(search_space)


    def get_normalised_pRef(self, pRef: PRef) -> PRef:
        min_fitness = np.min(pRef.fitness_array)
        normalised_fitnesses = pRef.fitness_array - min_fitness
        sum_fitness = np.sum(normalised_fitnesses)

        if sum_fitness == 0:
            raise Exception(f"The sum of fitnesses for {pRef} is 0, could not normalise")

        normalised_fitnesses /= sum_fitness

        return PRef(full_solutions=pRef.full_solutions,
                    fitness_array=normalised_fitnesses,  # this is the only thing that changes
                    full_solution_matrix=pRef.full_solution_matrix,
                    search_space=pRef.search_space)


    def get_benefit(self, ps: PS, normalised_pRef: PRef) -> float:
        return np.sum(normalised_pRef.fitnesses_of_observations(ps))

    def get_global_isolated_benefits(self, normalised_pRef: PRef) -> list[list[float]]:
        ss = normalised_pRef.search_space
        empty: PS = PS.empty(ss)
        def benefit_when_isolating(var: int, val: int) -> float:
            isolated = empty.with_fixed_value(var, val)
            return self.get_benefit(isolated, normalised_pRef)

        return [[benefit_when_isolating(var, val)
                 for val in range(ss.cardinalities[var])]
                for var in range(ss.amount_of_parameters)]

    def get_excluded(self, ps: PS):
        return ps.simplifications()


    def get_isolated_benefits(self, ps: PS,
                              global_isolated_benefits: list[list[float]]) -> ArrayOfFloats:
        return np.array([global_isolated_benefits[var][val]
                  for var, val in enumerate(ps.values)
                  if val != STAR])

    def get_excluded_benefits(self, ps: PS, normalised_pRef: PRef) -> ArrayOfFloats:
        return np.array([self.get_benefit(excluded, normalised_pRef) for excluded in self.get_excluded(ps)])

    def get_unnormalised_scores(self, pss: Iterable[PS], pRef: PRef) -> ArrayOfFloats:

        normalised_pRef = self.get_normalised_pRef(pRef)
        global_isolated_benefits = self.get_global_isolated_benefits(normalised_pRef)


        def get_single_score(ps: PS) -> float:
            pAB = self.get_benefit(ps, normalised_pRef)

            isolated = self.get_isolated_benefits(ps, global_isolated_benefits)
            excluded = self.get_excluded_benefits(ps, normalised_pRef)

            max_denominator = np.max(isolated * excluded)  # praying that they are always the same size!

            return pAB * np.log(pAB / max_denominator)


        return np.array([get_single_score(ps) for ps in pss])











