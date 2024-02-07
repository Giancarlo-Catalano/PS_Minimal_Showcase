from typing import Iterable

import numpy as np

import SearchSpace
from PRef import PRef
from PS import PS, STAR
from PSMetric.Metric import Metric
from custom_types import ArrayOfFloats


class KindaAtomicity(Metric):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "KindaAtomicity"



    def get_linkages_for_vars(self, pRef: PRef) -> np.ndarray:
        overall_avg_fitness = np.average(pRef.fitness_array)
        def chance_of_bta_for_ps(ps: PS):
            fitnesses = pRef.fitnesses_of_observations(ps)
            better_than_average = [fit for fit in fitnesses if fit > overall_avg_fitness]
            return len(better_than_average) / len(fitnesses)

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

            return sum((np.square(observed_distr)-np.square(expected_distr))/expected_distr)

        chi_squared_matrix = np.zeros((pRef.search_space.amount_of_parameters, pRef.search_space.amount_of_parameters))
        for var_a in range(pRef.search_space.amount_of_parameters):
            for var_b in range(var_a, pRef.search_space.amount_of_parameters):
                chi_squared_matrix[var_a][var_b] = chi_squared_between_vars(var_a, var_b)


        # then we mirror it for convenience...
        upper_triangle = np.triu(chi_squared_matrix, k=1)
        chi_squared_matrix = chi_squared_matrix + upper_triangle.T
        return chi_squared_matrix


    def get_unnormalised_scores(self, pss: list[PS], pRef: PRef) -> ArrayOfFloats:
        linkages = self.get_linkages_for_vars(pRef)
        return np.ones(len(pss), dtype=float)