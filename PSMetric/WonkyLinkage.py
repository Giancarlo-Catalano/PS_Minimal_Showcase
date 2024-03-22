from typing import TypeAlias, Optional

import numpy as np

import utils
from PRef import PRef
from PS import PS, STAR
from PSMetric.Metric import Metric
from custom_types import ArrayOfFloats

LinkageTable: TypeAlias = np.ndarray


class WonkyLinkage(Metric):
    linkage_table: Optional[LinkageTable]
    normalised_linkage_table: Optional[LinkageTable]

    def __init__(self):
        super().__init__()
        self.linkage_table = None
        self.normalised_linkage_table = None

    def __repr__(self):
        return "WonkyLinkage"

    def set_pRef(self, pRef: PRef):
        self.linkage_table = self.get_linkage_table(pRef)
        self.normalised_linkage_table = self.get_normalised_linkage_table(self.linkage_table)

    @staticmethod
    def get_linkage_table(pRef: PRef) -> LinkageTable:
        overall_average = np.average(pRef.fitness_array)

        def get_mean_of_ps(ps: PS) -> float:
            observations = pRef.fitnesses_of_observations(ps)
            if len(observations) == 0:
                return overall_average
            return np.average(observations)

        def variance(values: ArrayOfFloats) -> float:
            if len(values) < 2:
                return 0
            mean = np.average(values)
            return np.sum(np.square(values - mean)) / len(values) - 1

        def one_fixed_var(var, val) -> PS:
            return PS.empty(pRef.search_space).with_fixed_value(var, val)

        def two_fixed_vars(var_x, val_x, var_y, val_y) -> PS:
            return (PS.empty(pRef.search_space)
                    .with_fixed_value(var_x, val_x)
                    .with_fixed_value(var_y, val_y))


        levels = [list(range(cardinality)) for cardinality in pRef.search_space.cardinalities]

        def get_variance_of_variable(var: int) -> float:
            mean_fitnesses = np.array([get_mean_of_ps(one_fixed_var(var, val))
                                       for val in levels[var]])
            return variance(mean_fitnesses)

        def get_variance_for_pair_of_variables(var_x: int, var_y: int) -> float:
            mean_fitnesses = np.array([get_mean_of_ps(two_fixed_vars(var_x, val_x, var_y, val_y))
                                       for val_x in levels[var_x]
                                       for val_y in levels[var_y]])
            return variance(mean_fitnesses)

        def interaction_effect_between_vars(var_x: int, var_y: int) -> float:
            if var_x == var_y:
                return get_variance_of_variable(var_x)
            else:
                return get_variance_for_pair_of_variables(var_x, var_y)


        linkage_table = np.zeros((pRef.search_space.amount_of_parameters, pRef.search_space.amount_of_parameters))
        for var_a in range(pRef.search_space.amount_of_parameters):
            for var_b in range(var_a, pRef.search_space.amount_of_parameters):
                if var_a == var_b:
                    linkage = get_variance_of_variable(var_a)
                else:
                    linkage = get_variance_for_pair_of_variables(var_a, var_b)
                linkage_table[var_a][var_b] = linkage

        # then we mirror it for convenience...
        upper_triangle = np.triu(linkage_table, k=1)
        linkage_table = linkage_table + upper_triangle.T
        return linkage_table

    @staticmethod
    def get_normalised_linkage_table(linkage_table: LinkageTable):
        where_to_consider = np.triu(np.full_like(linkage_table, True, dtype=bool), k=0)
        triu_min = np.min(linkage_table, where=where_to_consider, initial=np.inf)
        triu_max = np.max(linkage_table, where=where_to_consider, initial=-np.inf)
        normalised_linkage_table: LinkageTable = (linkage_table - triu_min) / triu_max

        return normalised_linkage_table

    @staticmethod
    def get_quantized_linkage_table(linkage_table: LinkageTable):
        where_to_consider = np.triu(np.full_like(linkage_table, True, dtype=bool), k=1)
        average = np.average(linkage_table[where_to_consider])
        quantized_linkage_table: LinkageTable = np.array(linkage_table >= average, dtype=float)
        return quantized_linkage_table

    def get_linkage_scores(self, ps: PS) -> np.ndarray:
        fixed = ps.values != STAR
        fixed_combinations: np.array = np.outer(fixed, fixed)
        fixed_combinations = np.triu(fixed_combinations, k=1)
        return self.linkage_table[fixed_combinations]

    def get_normalised_linkage_scores(self, ps: PS) -> np.ndarray:
        fixed = ps.values != STAR
        fixed_combinations: np.array = np.outer(fixed, fixed)
        fixed_combinations = np.triu(fixed_combinations, k=0)
        return self.normalised_linkage_table[fixed_combinations]

    def get_single_score_using_avg(self, ps: PS) -> float:
        if ps.fixed_count() < 1:
            return 0
        else:
            return np.average(self.get_linkage_scores(ps))

    def get_single_normalised_score(self, ps: PS) -> float:
        self.used_evaluations += 1
        if ps.fixed_count() < 1:
            return 0
        else:
            return np.min(self.get_normalised_linkage_scores(ps))

    def get_single_score(self, ps: PS) -> float:
        return self.get_single_score_using_avg(ps)
