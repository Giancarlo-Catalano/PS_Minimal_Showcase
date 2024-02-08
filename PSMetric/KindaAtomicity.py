from typing import Iterable

import numpy as np

import SearchSpace
from PRef import PRef
from PS import PS, STAR
from PSMetric.Metric import Metric
from custom_types import ArrayOfFloats


from scipy.stats import chi2_contingency


class KindaAtomicity(Metric):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "KindaAtomicity"



    def get_linkages_for_vars(self, pRef: PRef) -> np.ndarray:
        overall_avg_fitness = np.average(pRef.fitness_array)
        bta_amount = np.sum(pRef.fitness_array>overall_avg_fitness, dtype=float)
        def chance_of_bta_for_ps(ps: PS) -> float:
            fitnesses = pRef.fitnesses_of_observations(ps)
            better_than_average = [fit for fit in fitnesses if fit > overall_avg_fitness]
            return len(better_than_average) / bta_amount

        empty = PS.empty(pRef.search_space)
        trivial_pss = [[empty.with_fixed_value(var_index, val)
                        for val in range(cardinality)]
                       for var_index, cardinality in enumerate(pRef.search_space.cardinalities)]
        def chance_of_bta_for_var(var_index: int) -> ArrayOfFloats:
            return np.array([chance_of_bta_for_ps(ps) for ps in trivial_pss[var_index]])


        distributions = [chance_of_bta_for_var(var) for var in range(pRef.search_space.amount_of_parameters)]


        def observed_bivariate_distribution(var_a: int, var_b: int) -> np.ndarray:
            return np.array([[chance_of_bta_for_ps(PS.merge(ps_a, ps_b))
                              for ps_b in trivial_pss[var_b]]
                             for ps_a in trivial_pss[var_a]])


        def chi_squared_between_vars(var_a: int, var_b: int) -> float:
            univariate_distr_a = distributions[var_a]
            univariate_distr_b = distributions[var_b]
            expected_distr: np.ndarray = univariate_distr_a.reshape((-1, 1)) * univariate_distr_b.reshape((1, -1))
            observed_distr: np.ndarray = observed_bivariate_distribution(var_a, var_b)

            expected_amounts = expected_distr * bta_amount
            observed_amounts = observed_distr * bta_amount

            chi2, p_value, dof, expected = chi2_contingency(observed_amounts)
            return chi2
            #return float(np.sum((np.square(observed_amounts)-np.square(expected_amounts))/expected_amounts))


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


        merges = np.array([[interaction_effect_between_pss(ps_a, ps_b) if PS.mergeable(ps_a, ps_b) else 0
                         for ps_b in flat_trivials]
                        for ps_a in flat_trivials])


        chi_squared_matrix = np.zeros((pRef.search_space.amount_of_parameters, pRef.search_space.amount_of_parameters))
        for var_a in range(pRef.search_space.amount_of_parameters-1):
            for var_b in range(var_a+1, pRef.search_space.amount_of_parameters):
                chi_squared_matrix[var_a][var_b] = interaction_effect_between_vars(var_a, var_b)


        # then we mirror it for convenience...
        upper_triangle = np.triu(chi_squared_matrix, k=1)
        chi_squared_matrix = chi_squared_matrix + upper_triangle.T
        return chi_squared_matrix


    def get_unnormalised_scores(self, pss: list[PS], pRef: PRef) -> ArrayOfFloats:
        linkages = self.get_linkages_for_vars(pRef)
        return np.ones(len(pss), dtype=float)