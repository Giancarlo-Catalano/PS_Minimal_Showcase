from FullSolution import FullSolution
from PRef import PRef
from PS import PS
from SearchSpace import SearchSpace
from utils import announce


class BenchmarkProblem:
    """ This is an interface for toy problems, which makes my code much prettier"""
    """ The main components of this class are:
     -  a search space: the combinatorial search space
     -  fitness_function: the fitness function to be MAXIMISED
     -  get_targets: the ideal PS catalog
     -  repr_pr: a way to represent the PS which makes sense for the problem (ie checkerboard would use a grid)
     
     A useful related class to look at is UnitaryProblem
     """
    search_space: SearchSpace

    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    def __repr__(self):
        raise Exception("An implementation of BenchmarkProblem does not implement __repr__")

    def get_reference_population(self, sample_size: int) -> PRef:
        with announce(f"Generating a uniform pRef with {sample_size} samples"):
            result = PRef.sample_from_search_space(search_space=self.search_space,
                                                 fitness_function=self.fitness_function,
                                                 amount_of_samples=sample_size)
        return result

    def repr_full_solution(self, fs: FullSolution) -> str:
        """default implementation"""
        return f"{fs}"

    def repr_ps(self, ps: PS) -> str:
        """default implementation"""
        return f"{ps}"

    def repr_fs(self, full_solution: FullSolution) -> str:
        return self.repr_ps(PS.from_FS(full_solution))

    def fitness_function(self, fs: FullSolution) -> float:
        raise Exception("An implementation of BenchmarkProblem does not implement the fitness function!!!")

    def get_targets(self) -> list[PS]:
        raise Exception("An implementation of BenchmarkProblem does not implement get_targets")

    def get_global_optima_fitness(self) -> float:
        raise Exception("An implementation of BenchmarkProblem does not implement get_global_optima_fitness")
