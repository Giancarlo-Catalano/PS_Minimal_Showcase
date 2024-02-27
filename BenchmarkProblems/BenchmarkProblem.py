from FullSolution import FullSolution
from PRef import PRef
from PS import PS
from SearchSpace import SearchSpace


class BenchmarkProblem:
    search_space: SearchSpace

    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    def __repr__(self):
        raise Exception("An implementation of BenchmarkProblem does not implement __repr__")


    def long_repr(self) -> str:
        """default implementation"""
        return self.__repr__()

    def get_pRef(self, sample_size: int) -> PRef:
        return PRef.sample_from_search_space(search_space=self.search_space,
                                             fitness_function=self.fitness_function,
                                             amount_of_samples=sample_size)

    def repr_full_solution(self, fs: FullSolution) -> str:
        """default behaviour"""
        return f"{fs}"

    def repr_ps(self, ps: PS) -> str:
        """default behaviour"""
        return f"{ps}"

    def fitness_function(self, fs: FullSolution) -> float:
        raise Exception("An implementation of BenchmarkProblem does not implement the fitness function!!!")

    def get_targets(self) -> list[PS]:
        raise Exception("An implementation of BenchmarkProblem does not implement get_targets")