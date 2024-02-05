import numpy as np


def remap_array_in_zero_one(input_array: np.ndarray):
    """remaps the values in the given array to be between 0 and 1"""
    min_value = np.min(input_array)
    max_value = np.max(input_array)

    if (min_value == max_value):
        return np.full(len(input_array), 0.5, dtype=float)  # all 0.5!

    return (input_array - min_value) / (max_value - min_value)