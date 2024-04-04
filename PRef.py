from typing import Iterable, Callable

import numpy as np

import utils
from EvaluatedFS import EvaluatedFS
from FullSolution import FullSolution
from PS import PS, STAR
from SearchSpace import SearchSpace
from custom_types import Fitness, ArrayOfFloats


class PRef:
    """
    This class represents the referenece population, and you should think of it as a list of solutions,
    and a list of their fitnesses. Everything else is just to make the calculations faster / easier to implement.
    """
    full_solutions: list[FullSolution]
    fitness_array: ArrayOfFloats
    full_solution_matrix: np.ndarray
    search_space: SearchSpace

    def __init__(self, full_solutions: Iterable[FullSolution],
                 fitness_array: Iterable[Fitness],
                 full_solution_matrix: np.ndarray,
                 search_space: SearchSpace):
        self.full_solutions = list(full_solutions)
        self.fitness_array = np.array(fitness_array)
        self.full_solution_matrix = full_solution_matrix
        self.search_space = search_space

    def __repr__(self):
        mean_fitness = np.average(self.fitness_array)
        return f"PRef with {len(self.full_solutions)} samples, mean = {mean_fitness:.2f}"

    @classmethod
    def from_full_solutions(cls, full_solutions: Iterable[FullSolution],
                            fitness_values: Iterable[Fitness],
                            search_space: SearchSpace):
        matrix = np.array([fs.values for fs in full_solutions])
        return cls(full_solutions, fitness_values, matrix, search_space)

    @classmethod
    def sample_from_search_space(cls, search_space: SearchSpace,
                                 fitness_function: Callable,
                                 amount_of_samples: int):
        samples = [FullSolution.random(search_space) for _ in range(amount_of_samples)]
        fitnesses = [fitness_function(fs) for fs in samples]
        return cls.from_full_solutions(samples, fitnesses, search_space)

    def fitnesses_of_observations(self, ps: PS) -> ArrayOfFloats:
        """
        This is the most important function of the class, and it roughly corresponds to the obs_PRef(ps) in the paper
        :param ps: a partial solution, where the * values are represented by -1
        :return: a list of floats, corresponding to the fitnesses of the observations of the ps
        within the reference population
        """
        remaining_rows = self.full_solution_matrix
        remaining_fitnesses = self.fitness_array

        for variable_index, variable_value in enumerate(ps.values):
            if variable_value != STAR:
                which_to_keep = remaining_rows[:, variable_index] == variable_value

                # update the current filtered results
                remaining_rows = remaining_rows[which_to_keep]
                remaining_fitnesses = remaining_fitnesses[which_to_keep]

        return remaining_fitnesses

    def fitnesses_of_observations_and_complement(self, ps: PS) -> (ArrayOfFloats, ArrayOfFloats):
        selected_rows = np.full(shape=self.fitness_array.shape, fill_value=True, dtype=bool)

        for variable_index, variable_value in enumerate(ps.values):
            if variable_value != STAR:
                rows_where_variable_matches = self.full_solution_matrix[:, variable_index] == variable_value
                selected_rows = np.logical_and(selected_rows, rows_where_variable_matches)

        return self.fitness_array[selected_rows], self.fitness_array[np.logical_not(selected_rows)]

    @property
    def sample_size(self) -> int:
        return len(self.fitness_array)

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
        where = np.logical_and(self.full_solution_matrix[:, var_a] == val_a,
                               self.full_solution_matrix[:, var_b] == val_b)
        return self.fitness_array[where]

    def get_evaluated_FSs(self) -> list[EvaluatedFS]:
        return [EvaluatedFS(full_solution=fs, fitness=fitness) for fs, fitness in
                zip(self.full_solutions, self.fitness_array)]

    def describe_self(self):
        min_fitness = np.min(self.fitness_array)
        max_fitness = np.max(self.fitness_array)
        avg_fitness = np.average(self.fitness_array)
        print(
            f"This PRef contains {self.sample_size} samples, where the minimum is {min_fitness}, the maximum = {max_fitness} and the average is {avg_fitness}")
