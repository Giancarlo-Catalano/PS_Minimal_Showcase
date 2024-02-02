from typing import Iterable

import numpy as np

import FullSolution
import HotEncoding
from custom_types import Fitness, BooleanMatrix, ArrayOfFloats
from SearchSpace import SearchSpace
from PS import PS

class PRef:
    full_solutions: list[FullSolution]
    fitness_array: ArrayOfFloats
    full_solution_matrix: BooleanMatrix
    search_space: SearchSpace

    def __init__(self, full_solutions: Iterable[FullSolution],
                 fitness_array: Iterable[Fitness],
                 full_solution_matrix: BooleanMatrix,
                 search_space: SearchSpace):
        self.full_solutions = list(full_solutions)
        self.fitness_array = np.array(fitness_array)
        self.full_solution_matrix = full_solution_matrix
        self.search_space = search_space


    @classmethod
    def from_full_solutions(cls, full_solutions: Iterable[FullSolution],
                            fitness_values: Iterable[Fitness],
                            search_space: SearchSpace):
        matrix = np.array(full_solutions)
        return cls(full_solutions, fitness_values, matrix, search_space)

    def fitnesses_of_observations(self, ps: PS) -> ArrayOfFloats:
        remaining_rows = self.full_solution_matrix
        remaining_fitnesses = self.fitness_array

        for variable_index, variable_value in enumerate(ps.values):
            if variable_value >= 0:
                which_to_keep = remaining_rows[:, variable_index] == variable_value

                # update the current filtered results
                remaining_rows = remaining_rows[which_to_keep]
                remaining_fitnesses = remaining_fitnesses[which_to_keep]

        return remaining_fitnesses


