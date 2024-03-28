from typing import Iterable, Callable

import numpy as np

import utils
from FullSolution import FullSolution
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

    def __repr__(self):
        mean_fitness = np.average(self.fitness_array)
        return f"PRef with {len(self.full_solutions)} samples, mean = {mean_fitness:.2f}"

    def long_repr(self):
        header_str = f"PRef with {len(self.full_solutions)} samples"

        fs_str = ""
        for fs, fitness in zip(self.full_solutions, self.fitness_array):
            fs_str += f"{fs}, fitness = {fitness}\n"

        matrix_str = f"The matrix has dimensions {self.full_solution_matrix.shape}"

        return "\n".join([header_str, fs_str, matrix_str])

    @classmethod
    def from_full_solutions(cls, full_solutions: Iterable[FullSolution],
                            fitness_values: Iterable[Fitness],
                            search_space: SearchSpace):
        matrix = np.array([fs.values for fs in full_solutions])
        return cls(full_solutions, fitness_values, matrix, search_space)

    @classmethod
    def from_just_matrix(cls, solution_matrix: np.ndarray,
                         fitness_array: Iterable[Fitness],
                         search_space: SearchSpace):
        return cls(full_solutions=[],  # dummy value
                   fitness_array=fitness_array,
                   full_solution_matrix=solution_matrix,
                   search_space=search_space)

    @classmethod
    def sample_from_search_space(cls, search_space: SearchSpace,
                                 fitness_function: Callable,
                                 amount_of_samples: int):
        samples = [FullSolution.random(search_space) for _ in range(amount_of_samples)]
        fitnesses = [fitness_function(fs) for fs in samples]
        return cls.from_full_solutions(samples, fitnesses, search_space)

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

    @property
    def sample_size(self) -> int:
        return len(self.fitness_array)


    @classmethod
    def concat(cls, a, b):
        new_full_solutions = a.full_solutions + b.full_solutions
        new_fitness_array = np.concatenate((a.fitness_array, b.fitness_array))
        new_full_solution_matrix = np.vstack((a.full_solution_matrix, b.full_solution_matrix))

        return cls(new_full_solutions, new_fitness_array, new_full_solution_matrix, a.search_space)

    def get_with_normalised_fitnesses(self):
        normalised_fitnesses = utils.remap_array_in_zero_one(self.fitness_array)
        return PRef(full_solutions=self.full_solutions,
                    fitness_array=normalised_fitnesses,  # this is the only thing that changes
                    full_solution_matrix=self.full_solution_matrix,
                    search_space=self.search_space)


    def get_fitnesses_matching_var_val(self, var: int, val: int) -> ArrayOfFloats:
        where = self.full_solution_matrix[:, var] == val
        return self.fitness_array[where]


    def get_fitnesses_matching_var_val_pair(self, var_a: int, val_a: int, var_b: int, val_b: int) -> ArrayOfFloats:
        where = np.logical_and(self.full_solution_matrix[:, var_a] == val_a, self.full_solution_matrix[:, var_b] == val_b)
        return self.fitness_array[where]

