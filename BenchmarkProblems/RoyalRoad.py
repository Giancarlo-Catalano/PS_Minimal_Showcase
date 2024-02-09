import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FullSolution import FullSolution
from SearchSpace import SearchSpace


class RoyalRoad(BenchmarkProblem):
    amount_of_cliques: int
    size_of_cliques: int

    def __init__(self, amount_of_cliques: int, size_of_cliques: int):
        self.amount_of_cliques = amount_of_cliques
        self.size_of_cliques = size_of_cliques
        amount_of_bits = self.amount_of_cliques * self.size_of_cliques
        super().__init__(SearchSpace([2 for _ in range(amount_of_bits)]))

    def __repr__(self):
        return (f"RR(amount_of_cliques = {self.amount_of_cliques}, "
                f"size_of_cliques = {self.size_of_cliques})")


    def unitary_fitness_function(self, amount_of_ones: int) -> float:
        if amount_of_ones == self.size_of_cliques:
            return 1.0
        else:
            return 0.0

    def fitness_function(self, fs: FullSolution) -> float:
        values = fs.values.copy()
        values = values.reshape((-1, self.size_of_cliques))
        unities = np.sum(values, axis=1)
        return sum(map(self.unitary_fitness_function, unities))

