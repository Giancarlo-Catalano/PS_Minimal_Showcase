"""
This class aims to build a linkage model from a PRef,
with the following requirements:
* does not require too many samples
* does not need for the PRef to be from a uniform distribution
"""
from typing import TypeAlias, Callable

import numpy as np
from scipy.stats import f_oneway

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PRef import PRef
from PS import PS
from SearchSpace import SearchSpace

from scipy.stats import pearsonr, spearmanr, kendalltau, kruskal, chi2_contingency

LinkageValue: TypeAlias = float
LinkageTable: TypeAlias = np.ndarray


def mirror_diagonal(linkage_table: LinkageTable) -> LinkageTable:
    """ This function takes a table that's only been filled in the upper triangle
    and returns a table that has those values mirrored around the diagonal"""
    upper_triangle = np.triu(linkage_table, k=1)
    return linkage_table + upper_triangle.T


def make_table_from_linkage_function(amount_of_parameters, linkage_function: Callable[[int, int], LinkageValue]):
    """
    Makes a linkage table by using the provided linkage function
    :param amount_of_parameters:
    :param linkage_function: a function f(x, y), which returns the linkage between variables x and y
    :return: a table, where entry [x, y] is the linkage calculated from f(x, y)
    """
    linkage_table = np.zeros((amount_of_parameters, amount_of_parameters))
    for var_a in range(amount_of_parameters):
        for var_b in range(var_a, amount_of_parameters):
            linkage_table[var_a][var_b] = linkage_function(var_a, var_b)

    return mirror_diagonal(linkage_table)


@utils.print_entry_and_exit
def get_linkage_table_alpha(pRef: PRef) -> LinkageTable:
    """ This version is the one used in the Linkage metric,
    and
        given two variables x and y,
        and their values a's and b's

    We can calculate the average fitness for each of those
        e.g the average fitness when the variable y is b
    and if we subtract the overall mean fitness we get the "benefit" of that value

    Then, if we take the combination x=a, y=b, we say
        benefit(x=a) = mean(x=a) - mean(*)
        benefit(y=b) = mean(y=a) - mean(*)

        benefit(x=a, y=b) = mean(x=a, y=b) - mean(*)

        expected_benefit(x=a, y=b) = benefit(x=a) + benefit(y=b)


        connection_term(x=a, y=b) = abs(expected_benefit(x=a, y=b) - benefit(x=a, y=b))

    And finally,
        linkage(x, y) = sum(connection_term(x=a, y=b)  for a in range(card[x]) for b in range(card[y]))
    """
    overall_average = np.average(pRef.fitness_array)

    def get_mean_benefit_of_ps(ps: PS):
        return np.average(pRef.fitnesses_of_observations(ps)) - overall_average

    def one_fixed_var(var, val) -> PS:
        return PS.empty(pRef.search_space).with_fixed_value(var, val)

    def two_fixed_vars(var_x, val_x, var_y, val_y) -> PS:
        return (PS.empty(pRef.search_space)
                .with_fixed_value(var_x, val_x)
                .with_fixed_value(var_y, val_y))

    marginal_benefits = [[get_mean_benefit_of_ps(one_fixed_var(var, val))
                          for val in range(cardinality)]
                         for var, cardinality in enumerate(pRef.search_space.cardinalities)]

    def interaction_effect_between_vars(var_x: int, var_y: int) -> float:
        def connection_term(val_a, val_b):
            expected_conditional = marginal_benefits[var_x][val_a] + marginal_benefits[var_y][val_b]
            observed_conditional = get_mean_benefit_of_ps(two_fixed_vars(var_x, val_a, var_y, val_b))

            return abs(expected_conditional - observed_conditional)

        cardinality_x = pRef.search_space.cardinalities[var_x]
        cardinality_y = pRef.search_space.cardinalities[var_y]
        return sum(connection_term(val_a, val_b)
                   for val_a in range(cardinality_x)
                   for val_b in range(cardinality_y))

    return make_table_from_linkage_function(pRef.search_space.amount_of_parameters,
                                            linkage_function=interaction_effect_between_vars)


def mutual_information_linkage_table(search_space: SearchSpace, get_p) -> LinkageTable:
    def one_fixed_var(var, val) -> PS:
        return PS.empty(search_space).with_fixed_value(var, val)

    def two_fixed_vars(var_x, val_x, var_y, val_y) -> PS:
        return (PS.empty(search_space)
                .with_fixed_value(var_x, val_x)
                .with_fixed_value(var_y, val_y))

    marginal_probabilities = [[get_p(one_fixed_var(var, val))
                               for val in range(cardinality)]
                              for var, cardinality in enumerate(search_space.cardinalities)]

    def interaction_effect_between_vars(var_x: int, var_y: int) -> float:
        """Returns the chi squared value between x and y"""

        def connection_term(val_a, val_b):
            p_x_y = get_p(two_fixed_vars(var_x, val_a, var_y, val_b))
            p_x = marginal_probabilities[var_x][val_a]
            p_y = marginal_probabilities[var_y][val_b]
            return p_x_y * np.log(p_x_y / (p_x * p_y))

        cardinality_x = search_space.cardinalities[var_x]
        cardinality_y = search_space.cardinalities[var_y]
        return sum(connection_term(val_a, val_b)
                   for val_a in range(cardinality_x)
                   for val_b in range(cardinality_y))

    return make_table_from_linkage_function(search_space.amount_of_parameters,
                                            linkage_function=interaction_effect_between_vars)


@utils.print_entry_and_exit
def get_linkage_table_beta(pRef: PRef) -> LinkageTable:
    overall_median = np.median(pRef.fitness_array)

    def get_p(ps: PS):
        observations = pRef.fitnesses_of_observations(ps)
        return len([o for o in observations if o >= overall_median]) / len(observations)

    return mutual_information_linkage_table(pRef.search_space, get_p)


@utils.print_entry_and_exit
def get_linkage_table_gamma(pRef: PRef) -> LinkageTable:
    overall_mean = np.average(pRef.fitness_array)

    def get_p(ps: PS):
        observations = pRef.fitnesses_of_observations(ps)
        return len([o for o in observations if o >= overall_mean]) / len(observations)

    return mutual_information_linkage_table(pRef.search_space, get_p)


def truncation_selection_pRef(pRef: PRef) -> PRef:
    items = list(zip(pRef.full_solutions, pRef.fitness_array))
    items.sort(key=utils.second, reverse=True)

    amount_to_keep = len(items) // 2
    items = items[:amount_to_keep]
    full_solutions, fitness_values = utils.unzip(items)
    return PRef.from_full_solutions(full_solutions=full_solutions,
                                    fitness_values=fitness_values,
                                    search_space=pRef.search_space)
@utils.print_entry_and_exit
def use_truncated_pRef(linkage_table_generating_function: Callable[[PRef], LinkageTable]) -> Callable[
    [PRef], LinkageTable]:


    def apply(pRef: PRef):
        truncated_pRef = truncation_selection_pRef(pRef)
        return linkage_table_generating_function(truncated_pRef)

    return apply


def calculate_linkage_using_statistics(solutions, test: str):
    num_variables = solutions.shape[1]
    linkage_matrix = np.zeros((num_variables, num_variables))

    for i in range(num_variables):
        for j in range(i+1, num_variables):
            var_i_values = solutions[:, i]
            var_j_values = solutions[:, j]

            if test == 'pearson':
                correlation, p_value = pearsonr(var_i_values, var_j_values)
            elif test == 'anova':
                correlation, p_value = f_oneway(var_i_values, var_j_values)
            elif test == 'spearman':
                correlation, p_value = spearmanr(var_i_values, var_j_values)
            elif test == 'kendall':
                correlation, p_value = kendalltau(var_i_values, var_j_values)
            elif test == 'kruskal':
                correlation, p_value = kruskal(var_i_values, var_j_values)
            elif test == 'chi_square':
                contingency_table = np.array([var_i_values, var_j_values])
                correlation, p_value, _, _ = chi2_contingency(contingency_table)
            else:
                raise ValueError(
                    "Invalid test type. Choose from 'pearson', 'spearman', 'kendall', 'kruskal', or 'chi_square'.")

            linkage_matrix[i, j] = correlation

    return mirror_diagonal(linkage_matrix)


def linkage_using_method(test: str):
    def apply(pRef: PRef):
        truncated_pRef = truncation_selection_pRef(pRef)
        solutions = truncated_pRef.full_solution_matrix

        return calculate_linkage_using_statistics(solutions, test = test)

    return apply


def test_linkage_tables(benchmark_problem: BenchmarkProblem):
    sample_sizes = [10000]
    methods = {"original": get_linkage_table_alpha,
               "truncated_marginal_median": use_truncated_pRef(get_linkage_table_beta),
               "truncated_marginal_mean": use_truncated_pRef(get_linkage_table_gamma)}

    methods.update({method: linkage_using_method(method)
                    for method in {"pearson", "anova", "spearman", "kendall", "kruskal"}})

    for sample_size in sample_sizes:
        pRef = benchmark_problem.get_pRef(sample_size)
        for method_key in methods:
            function = methods[method_key]
            linkage_table = function(pRef)
            linkage_table = np.nan_to_num(linkage_table, nan=0)
            print("Finished obtaining the table, hopefully you're debugging")
