from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FullSolution import FullSolution
from PS import PS
from SearchSpace import SearchSpace


class OneMax(BenchmarkProblem):
    amount_of_bits: int

    def __init__(self, amount_of_bits: int):
        self.amount_of_bits = amount_of_bits
        super().__init__(SearchSpace([2 for _ in range(self.amount_of_bits)]))

    def fitness_function(self, fs: FullSolution) -> float:
        return fs.values.sum(dtype=float)

    def get_targets(self) -> list[PS]:
        empty = PS.empty(self.search_space)
        return [empty.with_fixed_value(variable_position=var, fixed_value=1) for var in range(self.amount_of_bits)]


    def __repr__(self):
        return f"OneMax({self.amount_of_bits})"
    def get_global_optima_fitness(self) -> float:
        return float(self.amount_of_bits)