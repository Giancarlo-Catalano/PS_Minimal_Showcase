from BenchmarkProblems.UnitaryProblem import UnitaryProblem


class OneMax(UnitaryProblem):

    def __init__(self, amount_of_cliques: int, clique_size: int):
        super().__init__(amount_of_cliques, clique_size)

    def get_problem_name(self) -> str:
        return "OneMax"

    @staticmethod
    def unitary_function(self, bitcount: int) -> float:
        return float(bitcount)
