from typing import Any

import numpy as np


def first(pair: (Any, Any)) -> Any:
    return pair[0]


def second(pair: (Any, Any)) -> Any:
    return pair[1]



def unzip(zipped):
    if (len(zipped) == 0):
        return []

    group_amount = len(zipped[0])

    def get_nth_group(n):
        return [elem[n] for elem in zipped]

    return tuple(get_nth_group(n) for n in range(group_amount))


def remap_array_in_zero_one(input_array: np.ndarray):
    """remaps the values in the given array to be between 0 and 1"""
    min_value = np.min(input_array)
    max_value = np.max(input_array)

    if (min_value == max_value):
        return np.full(len(input_array), 0.5, dtype=float)  # all 0.5!

    return (input_array - min_value) / (max_value - min_value)



def remap_each_column_in_zero_one(input_matrix: np.ndarray) -> np.ndarray:
    min_for_each_column = np.min(input_matrix, axis=0)
    result_matrix = input_matrix-min_for_each_column
    max_for_each_column = np.max(result_matrix, axis=0)
    result_matrix /= max_for_each_column
    return result_matrix
