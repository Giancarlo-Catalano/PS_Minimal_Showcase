from enum import Enum, auto

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.Checkerboard import CheckerBoard
from BenchmarkProblems.OneMax import OneMax
from BenchmarkProblems.ParityProblem import ParityProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from FullSolution import FullSolution
from SearchSpace import SearchSpace
from custom_types import ArrayOfInts


class ToyProblem(Enum):
    CONSTANT = auto()
    ONEMAX = auto()
    ROYALROAD = auto()
    TRAPK = auto()
    PARITY = auto()

    def __repr__(self):
        return "CORTP"[self.value-1]

    def __str__(self):
        return "CORTP"[self.value-1]


def toy_enum_from_string(string: str) -> ToyProblem:
    match string:
        case "O":
            return ToyProblem.ONEMAX
        case "R":
            return ToyProblem.ROYALROAD
        case "T":
            return ToyProblem.TRAPK
        case "P":
            return ToyProblem.PARITY
        case _:
            return ToyProblem.CONSTANT


class ToyAmalgam(BenchmarkProblem):
    problems = list[ToyProblem]
    clique_size = int

    def __init__(self, toy_problems: str, clique_size: int):
        self.problems = [toy_enum_from_string(s) for s in toy_problems]
        self.clique_size = clique_size
        super().__init__(SearchSpace([2 for _ in range(self.amount_of_bits)]))

    @property
    def amount_of_bits(self) -> int:
        return len(self.problems) * self.clique_size

    def fitness_for_clique_bitcount(self, bitcount: int, problem: ToyProblem) -> float:
        match problem:
            case ToyProblem.CONSTANT:
                return float(self.clique_size)
            case ToyProblem.ONEMAX:
                return OneMax.unitary_function(bitcount, self.clique_size)
            case ToyProblem.ROYALROAD:
                return RoyalRoad.unitary_function(bitcount, self.clique_size)
            case ToyProblem.TRAPK:
                return Trapk.unitary_function(bitcount, self.clique_size)
            case ToyProblem.PARITY:
                return ParityProblem.unitary_function(bitcount, self.clique_size)

    def get_bit_counts(self, full_solution: FullSolution) -> ArrayOfInts:
        bits = full_solution.values.reshape((-1, self.clique_size))
        return np.sum(bits, axis=1)
    def fitness_function(self, fs: FullSolution) -> float:
        bit_counts = self.get_bit_counts(fs)
        return sum(self.fitness_for_clique_bitcount(bc, problem) for bc, problem in zip(bit_counts, self.problems))

    def __repr__(self):
        return f"ToyAmalgam({''.join(f'{p}' for p in self.problems)}, clique size = {self.clique_size}"
