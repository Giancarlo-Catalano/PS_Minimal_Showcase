from BenchmarkProblems.UnitaryProblem import UnitaryProblem


class Trapk(UnitaryProblem):

    def __init__(self, amount_of_cliques: int, clique_size: int):
        super().__init__(amount_of_cliques, clique_size)

    def get_problem_name(self) -> str:
        return "Trapk"


    @staticmethod
    def unitary_function(bitcount: int, clique_size: int) -> float:
        if bitcount == clique_size:
            return float(clique_size)
        else:
            return float(clique_size - bitcount - 1)