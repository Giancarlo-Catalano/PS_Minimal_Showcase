from BenchmarkProblems.UnitaryProblem import UnitaryProblem


class ParityProblem(UnitaryProblem):
    def __init__(self, amount_of_cliques: int, clique_size: int):
        super().__init__(amount_of_cliques, clique_size)

    def get_problem_name(self) -> str:
        return "ParityProblem"

    @staticmethod
    def unitary_function(bitcount: int, clique_size: int) -> float:
        if bitcount % 2 == 1:
            return float(clique_size)
        else:
            return 0.0