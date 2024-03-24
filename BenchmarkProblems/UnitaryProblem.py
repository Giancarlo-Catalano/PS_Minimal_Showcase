import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FullSolution import FullSolution
from PS import PS
from SearchSpace import SearchSpace
from custom_types import ArrayOfInts


class UnitaryProblem(BenchmarkProblem):
    amount_of_cliques: int
    clique_size: int

    def __init__(self,
                 amount_of_cliques: int,
                 clique_size: int):
        self.amount_of_cliques = amount_of_cliques
        self.clique_size = clique_size
        search_space = SearchSpace([2 for _ in range(amount_of_cliques * clique_size)])
        super().__init__(search_space)

    def get_bit_counts(self, full_solution: FullSolution) -> ArrayOfInts:
        bits = full_solution.values.reshape((-1, self.clique_size))
        return np.sum(bits, axis=1)

    def unitary_function(self, bitcount: int) -> float:
        raise Exception("An implementation of UnitaryProblem does not implement .unitary_function")

    def get_optimal_clique(self) -> FullSolution:
        raise Exception("An implementation of UnitaryProblem does not implement get_optimal_clique")

    def get_optimal_fitness_per_clique(self) -> float:
        optimal_clique = self.get_optimal_clique()
        bitcount = optimal_clique.values.sum()
        return self.unitary_function(bitcount)

    def get_global_optima_fitness(self) -> float:
        return self.get_optimal_fitness_per_clique() * self.amount_of_cliques

    def fitness(self, full_solution: FullSolution) -> float:
        return sum(self.unitary_function(bc) for bc in self.get_bit_counts(full_solution))

    def __repr__(self):
        raise Exception("An implementation of UnitaryProblem does not implement .__repr__")

    def repr_ps(self, ps: PS) -> str:
        def repr_cell(cell_value: int) -> str:
            match cell_value:
                case 0:
                    return "0"
                case 1:
                    return "1"
                case _:
                    return "*"

        def repr_clique(clique: np.ndarray) -> str:
            return f'[{" ".join(repr_cell(cell) for cell in clique)}]'

        cliques = ps.values.reshape((-1, self.clique_size))
        return " ".join(repr_clique(clique) for clique in cliques)
