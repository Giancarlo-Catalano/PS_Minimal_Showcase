from typing import TypeAlias, Optional

import numpy as np

import SearchSpace
from PRef import PRef
from PS import PS, STAR
from PSMetric.Metric import Metric
from custom_types import ArrayOfFloats

LinkageTable: TypeAlias = np.ndarray


class LinkageViaMeanFitDiff(Metric):
    linkage_table: Optional[LinkageTable]
    worst_linkage: Optional[float]

    def __init__(self):
        super().__init__()
        self.linkage_table = None
        self.worst_linkage = None

    def __repr__(self):
        return "KindaAtomicity"

    def set_pRef(self, pRef: PRef):
        self.linkage_table = self.get_linkage_table(pRef)
        self.worst_linkage = self.linkage_table.max(initial=0)

    @staticmethod
    def get_linkage_table(pRef: PRef) -> LinkageTable:
        overall_avg_fitness = np.average(pRef.fitness_array)

        empty = PS.empty(pRef.search_space)
        trivial_pss = [[empty.with_fixed_value(var_index, val)
                        for val in range(cardinality)]
                       for var_index, cardinality in enumerate(pRef.search_space.cardinalities)]

        def interaction_effect_between_pss(ps_a, ps_b) -> float:
            mean_a = np.average(pRef.fitnesses_of_observations(ps_a))
            mean_b = np.average(pRef.fitnesses_of_observations(ps_b))
            mean_both = np.average(pRef.fitnesses_of_observations(PS.merge(ps_a, ps_b)))

            benefit_a = mean_a - overall_avg_fitness
            benefit_b = mean_b - overall_avg_fitness
            benefit_both = mean_both - overall_avg_fitness
            return abs(benefit_both - benefit_a - benefit_b)

        def interaction_effect_between_vars(var_a: int, var_b: int) -> float:
            return sum([interaction_effect_between_pss(ps_a, ps_b)
                        for ps_b in trivial_pss[var_b]
                        for ps_a in trivial_pss[var_a]])

        linkage_table = np.zeros((pRef.search_space.amount_of_parameters, pRef.search_space.amount_of_parameters))
        for var_a in range(pRef.search_space.amount_of_parameters - 1):
            for var_b in range(var_a + 1, pRef.search_space.amount_of_parameters):
                linkage_table[var_a][var_b] = interaction_effect_between_vars(var_a, var_b)

        # then we mirror it for convenience...
        upper_triangle = np.triu(linkage_table, k=1)
        linkage_table = linkage_table + upper_triangle.T
        return linkage_table

    def get_linkage_scores(self, ps: PS) -> np.array:
        fixed = ps.values != STAR
        fixed_combinations: np.array = np.outer(fixed, fixed)
        np.fill_diagonal(fixed_combinations, False)  # remove relexive combinations
        return self.linkage_table[fixed_combinations]

    def get_single_score(self, ps: PS) -> float:
        if ps.fixed_count() < 2:
            return 0
        else:
            fixed = ps.values != STAR
            fixed_combinations: np.array = np.outer(fixed, fixed)
            np.fill_diagonal(fixed_combinations, False)  # remove reflexive combinations
            return np.min(self.linkage_table, where=fixed_combinations, initial=self.worst_linkage)


class SimplerAtomicity(Metric):
    normalised_pRef: Optional[PRef]
    global_isolated_benefits: Optional[list[list[float]]]

    def __init__(self):
        super().__init__()
        self.normalised_pRef = None
        self.global_isolated_benefits = None

    def set_pRef(self, pRef: PRef):
        self.normalised_pRef = self.get_normalised_pRef(pRef)
        self.global_isolated_benefits = self.get_global_isolated_benefits()

    def __repr__(self):
        return "SimplerAtomicity"

    def get_isolated_in_search_space(self, search_space: SearchSpace.SearchSpace) -> list[PS]:
        empty: PS = PS.empty(search_space)
        return empty.specialisations(search_space)

    def get_normalised_pRef(self, pRef: PRef) -> PRef:
        min_fitness = np.min(pRef.fitness_array)
        normalised_fitnesses = pRef.fitness_array - min_fitness
        sum_fitness = np.sum(normalised_fitnesses, dtype=float)

        if sum_fitness == 0:
            raise Exception(f"The sum of fitnesses for {pRef} is 0, could not normalise")

        # normalised_fitnesses /= sum_fitness

        return PRef(full_solutions=pRef.full_solutions,
                    fitness_array=normalised_fitnesses,  # this is the only thing that changes
                    full_solution_matrix=pRef.full_solution_matrix,
                    search_space=pRef.search_space)

    def get_benefit(self, ps: PS) -> float:
        return float(np.sum(self.normalised_pRef.fitnesses_of_observations(ps)))

    def get_global_isolated_benefits(self) -> list[list[float]]:
        """uses self.normalised_pRef"""
        ss = self.normalised_pRef.search_space
        empty: PS = PS.empty(ss)

        def benefit_when_isolating(var: int, val: int) -> float:
            isolated = empty.with_fixed_value(var, val)
            return self.get_benefit(isolated)

        return [[benefit_when_isolating(var, val)
                 for val in range(ss.cardinalities[var])]
                for var in range(ss.amount_of_parameters)]

    def get_isolated_benefits(self, ps: PS) -> ArrayOfFloats:
        return np.array([self.global_isolated_benefits[var][val]
                         for var, val in enumerate(ps.values)
                         if val != STAR])

    def get_excluded_benefits(self, ps: PS) -> ArrayOfFloats:
        exclusions = ps.simplifications()
        return np.array([self.get_benefit(excluded) for excluded in exclusions])

    def get_single_score(self, ps: PS):
        pAB = self.get_benefit(ps)
        if pAB == 0.0:
            return pAB

        isolated = self.get_isolated_benefits(ps)
        excluded = self.get_excluded_benefits(ps)

        if len(isolated) == 0:  # ie we have the empty ps
            return 0

        max_denominator = np.max(isolated * excluded)  # praying that they are always the same size!

        result = pAB / max_denominator  # with no logs or anything
        if np.isnan(result).any():
            raise Exception("There is a nan value returned in atomicity")
        return result
