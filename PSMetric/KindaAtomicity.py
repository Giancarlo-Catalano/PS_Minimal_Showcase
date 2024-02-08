from typing import Iterable, TypeAlias

import numpy as np

import SearchSpace
from PRef import PRef
from PS import PS, STAR
from PSMetric.Metric import Metric
from custom_types import ArrayOfFloats

LinkageTable: TypeAlias = np.ndarray

class KindaAtomicity(Metric):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "KindaAtomicity"



    def get_linkages_for_vars(self, pRef: PRef) -> LinkageTable:
        overall_avg_fitness = np.average(pRef.fitness_array)

        empty = PS.empty(pRef.search_space)
        trivial_pss = [[empty.with_fixed_value(var_index, val)
                        for val in range(cardinality)]
                       for var_index, cardinality in enumerate(pRef.search_space.cardinalities)]

        # debug
        flat_trivials = [var_group[n] for n in [0, 1] for var_group in trivial_pss] # [ps for var_group in trivial_pss for ps in var_group]

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
        for var_a in range(pRef.search_space.amount_of_parameters-1):
            for var_b in range(var_a+1, pRef.search_space.amount_of_parameters):
                linkage_table[var_a][var_b] = interaction_effect_between_vars(var_a, var_b)


        # then we mirror it for convenience...
        upper_triangle = np.triu(linkage_table, k=1)
        linkage_table = linkage_table + upper_triangle.T
        return linkage_table


    def get_linkage_scores(self, ps: PS, linkage_table: LinkageTable) -> np.array:
        fixed = ps.values != STAR
        fixed_combinations: np.array = np.outer(fixed, fixed)
        np.fill_diagonal(fixed_combinations, False)  # remove relexive combinations
        return linkage_table[fixed_combinations]

    def get_minimum_linkage_value(self, ps: PS, linkage_table: LinkageTable, otherwise:float) -> float:
        if ps.fixed_count() < 2:
            return 0
        else:
            fixed = ps.values != STAR
            fixed_combinations: np.array = np.outer(fixed, fixed)
            np.fill_diagonal(fixed_combinations, False)  # remove reflexive combinations
            return np.min(linkage_table, where = fixed_combinations, initial=otherwise)



    def get_unnormalised_scores(self, pss: list[PS], pRef: PRef) -> ArrayOfFloats:
        linkage_table = self.get_linkages_for_vars(pRef)
        worst = linkage_table.max(initial=0)
        return np.array([self.get_minimum_linkage_value(ps, linkage_table, otherwise=worst) for ps in pss])