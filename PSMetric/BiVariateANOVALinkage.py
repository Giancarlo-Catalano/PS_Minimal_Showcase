import itertools
from typing import TypeAlias, Optional

import numpy as np

import utils
from PRef import PRef
from PS import PS, STAR
from PSMetric.Linkage import Linkage
from PSMetric.Metric import Metric
from scipy.stats import f

LinkageTable: TypeAlias = np.ndarray


class BiVariateANOVALinkage(Metric):
    linkage_table: Optional[LinkageTable]
    normalised_linkage_table: Optional[LinkageTable]

    def __init__(self):
        super().__init__()
        self.linkage_table = None
        self.normalised_linkage_table = None

    def __repr__(self):
        return "BiVariateANOVALinkage"

    def set_pRef(self, pRef: PRef):
        print("Calculating linkages...", end="")
        self.linkage_table = self.get_linkage_table(pRef)
        self.normalised_linkage_table = Linkage.get_quantized_linkage_table(self.linkage_table)
        print("Finished")
        # self.normalised_linkage_table = self.get_normalised_linkage_table(self.linkage_table)

    def get_ANOVA_interaction_table_old(self, pRef: PRef) -> LinkageTable:
        """every entry in this table will be a p-value, so in theory smaller values have stronger linkage"""

        def interaction_test(data: np.ndarray, fitnesses: np.ndarray):
            # Perform a 2-factor ANOVA test with interaction term
            grand_mean = np.mean(fitnesses)
            dof_total = len(fitnesses) - 1
            n = data.shape[0]

            # Calculate sums of squares for factors
            sum_sq_factor1 = np.sum(
                (np.mean(fitnesses[data[:, 0] == level]) - grand_mean) ** 2 for level in np.unique(data[:, 0]))
            sum_sq_factor2 = np.sum(
                (np.mean(fitnesses[data[:, 1] == level]) - grand_mean) ** 2 for level in np.unique(data[:, 1]))
            sum_sq_interaction = np.sum((np.mean(fitnesses[(data[:, 0] == level[0]) & (data[:, 1] == level[1])]) -
                                         np.mean(fitnesses[data[:, 0] == level[0]]) -
                                         np.mean(fitnesses[data[:, 1] == level[1]]) +
                                         grand_mean) ** 2 for level in
                                        itertools.product(np.unique(data[:, 0]), np.unique(data[:, 1])))

            # Calculate error sum of squares
            ss_error = np.sum((fitnesses - np.mean(fitnesses)) ** 2)

            # Calculate degrees of freedom
            dof_factor1 = len(np.unique(data[:, 0])) - 1
            dof_factor2 = len(np.unique(data[:, 1])) - 1
            dof_interaction = dof_factor1 * dof_factor2
            dof_error = dof_total - (dof_factor1 + dof_factor2 + dof_interaction)

            # Calculate mean squares
            ms_factor1 = sum_sq_factor1 / dof_factor1
            ms_factor2 = sum_sq_factor2 / dof_factor2
            ms_interaction = sum_sq_interaction / dof_interaction
            ms_error = ss_error / dof_error

            # Calculate F statistic
            f_statistic = (ms_interaction / ms_error) if ms_error != 0 else np.inf

            # Calculate p-value
            p_value = 1 - f.cdf(f_statistic, dof_interaction, dof_error)
            return p_value

        def calculate_interaction(data: np.ndarray, fitnesses: np.ndarray):
            num_features = data.shape[1]
            interaction_table = np.zeros((num_features, num_features))
            for i, j in itertools.combinations(range(num_features), 2):
                interaction_data = np.column_stack((data[:, i], data[:, j], data[:, i] * data[:, j]))
                interaction_table[i, j] = interaction_test(interaction_data, fitnesses)
            return interaction_table + interaction_table.T  # Make the table symmetric

        solutions = pRef.full_solution_matrix
        fitnesses = pRef.fitness_array

        return calculate_interaction(solutions, fitnesses)

    def get_ANOVA_interaction_table(self, pRef: PRef) -> LinkageTable:
        """every entry in this table will be a p-value, so in theory smaller values have stronger linkage"""
        solutions = pRef.full_solution_matrix
        fitnesses = pRef.fitness_array

        grand_mean = np.mean(fitnesses)
        n = pRef.sample_size
        dof_total = n - 1

        levels = pRef.search_space.cardinalities

        def interaction_test(i: int, j: int):
            """ Perform a 2-factor ANOVA test with interaction term """

            values_i = solutions[:, i]
            values_j = solutions[:, j]

            levels_i = range(levels[i])
            levels_j = range(levels[j])

            where_values_i = [(values_i == level_i) for level_i in levels_i]
            where_values_j = [(values_j == level_j) for level_j in levels_j]

            # Calculating the sum of squares for the interaction
            # (Normally we'd also calculate the marginal sum of squares, but we don't need them here.

            sum_sq_interaction = np.sum((np.mean(fitnesses[where_val_i & where_val_j]) -
                                         np.mean(fitnesses[where_val_i]) -
                                         np.mean(fitnesses[where_val_j]) +
                                         grand_mean) ** 2 for where_val_i, where_val_j in
                                        itertools.product(where_values_i, where_values_j))

            # Calculate error sum of squares
            ss_error = np.sum((fitnesses - np.mean(fitnesses)) ** 2)

            # Calculate degrees of freedom
            dof_factor_i = pRef.search_space.cardinalities[i] - 1
            dof_factor_j = pRef.search_space.cardinalities[j] - 1
            dof_interaction = dof_factor_i * dof_factor_j
            dof_error = dof_total - (dof_factor_i + dof_factor_j + dof_interaction)

            # Calculate mean squares
            ms_interaction = sum_sq_interaction / dof_interaction
            ms_error = ss_error / dof_error

            # Calculate F statistic
            f_statistic = (ms_interaction / ms_error) if ms_error != 0 else np.inf

            # Calculate p-value
            p_value = 1 - f.cdf(f_statistic, dof_interaction, dof_error)
            return p_value

        def calculate_interaction(data: np.ndarray):
            num_features = data.shape[1]
            interaction_table = np.zeros((num_features, num_features))
            for i, j in itertools.combinations(range(num_features), 2):
                interaction_table[i, j] = interaction_test(i, j)
            return interaction_table + interaction_table.T  # Make the table symmetric

        return calculate_interaction(solutions)

    def get_linkage_table(self, pRef: PRef):
        # from_anova = self.get_ANOVA_interaction_table(pRef)
        table = 1 - self.get_ANOVA_interaction_table(pRef)
        # for debugging purposes, just so that it looks prettier in the PyCharm debugging window.
        np.fill_diagonal(table, 0)
        return table


    def get_normalised_linkage_scores(self, ps: PS) -> np.ndarray:
        fixed = ps.values != STAR
        fixed_combinations: np.array = np.outer(fixed, fixed)
        fixed_combinations = np.triu(fixed_combinations, k=1)
        return self.normalised_linkage_table[fixed_combinations]

    def get_single_normalised_score(self, ps: PS) -> float:
        self.used_evaluations += 1
        if ps.fixed_count() < 2:
            return 0
        else:
            return np.min(self.get_normalised_linkage_scores(ps))
