from typing import Iterable

import numpy as np

import FullSolution
from custom_types import BooleanMatrix
import PS
from SearchSpace import SearchSpace


def encode_full_solutions(full_solutions: Iterable[FullSolution],
                          search_space: SearchSpace) -> BooleanMatrix:
    resulting_matrix = np.zeros((len(full_solutions), search_space.hot_encoded_length), dtype=bool)

    full_solution_value_matrix = np.array([full_solution.values for full_solution in full_solutions])

    for variable_index in search_space.amount_of_parameters:
        for value in search_space.cardinalities[variable_index]:
            # this should be going in the expected order, 0, 1, 2, 3 etc.. but this is just to make sure
            column = search_space.precomputed_offsets[variable_index]+value
            resulting_matrix[:, column] = full_solution_value_matrix[:, variable_index] == value

    return resulting_matrix


def encode_partial_solutions(partial_solutions: Iterable[PS],
                             search_space: SearchSpace) -> BooleanMatrix:
    # this piece of code is identical to encode_full_solutions, at least at the moment
    resulting_matrix = np.zeros((len(partial_solutions), search_space.hot_encoded_length), dtype=bool)

    partial_solution_matrix = np.array([ps.values for ps in partial_solutions])

    for variable_index in range(search_space.amount_of_parameters):
        for value in search_space.cardinalities[variable_index]:
            # this should be going in the expected order, 0, 1, 2, 3 etc.. but this is just to make sure
            column = search_space.precomputed_offsets[variable_index] + value
            resulting_matrix[:, column] = partial_solution_matrix[:, variable_index] == value

    return resulting_matrix