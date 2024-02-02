from typing import Iterable

import numpy as np
from bitarray import bitarray, frozenbitarray

from FullSolution import FullSolution
from SearchSpace import SearchSpace
from custom_types import ArrayOfInts


class PS:
    values: ArrayOfInts

    def __init__(self, values: Iterable[int]):
        self.values = np.fromiter(values, dtype=int)

    def __len__(self):
        return len(self.values)

    def __repr__(self) -> str:
        def repr_single(cell_value: int) -> str:
            return f'{cell_value}' if cell_value >=0 else '*'

        return "[" + "".join(map(repr_single, self.values))

    @classmethod
    def empty(cls, search_space: SearchSpace):
        values = np.full(search_space.amount_of_parameters, -1)

    def is_fully_fixed(self) -> bool:
        return np.all(self.values == -1)

    def to_FS(self) -> FullSolution:
        return FullSolution(self.values)

    @classmethod
    def from_FS(cls, fs: FullSolution):
        return cls(fs.values)

    def with_unfixed_value(self, variable_position: int):
        new_values = np.copy(self.values)
        new_values[variable_position] = -1
        return PS(new_values)

    def with_fixed_value(self, variable_position: int, fixed_value: int):
        new_values = np.copy(self.values)
        new_values[variable_position] = fixed_value
        return PS(new_values)

    def get_fixed_variable_positions(self) -> list[int]:
        return [position for position, value in enumerate(self.values) if value >= 0]

    def get_unfixed_variable_positions(self) -> list[int]:
        return [position for position, value in enumerate(self.values) if value == -1]

    def simplifications(self):
        return [self.with_unfixed_value(i) for i in self.get_fixed_variable_positions()]

    def specialisations(self, search_space: SearchSpace):
        return [self.with_fixed_value(position, value)
                for position in self.get_unfixed_variable_positions()
                for value in range(search_space.cardinalities[position])]

    def __eq__(self, other) -> bool:
        """ This had to be optimised"""
        return np.array_equal(self.values, other.values)


    @classmethod
    def mergeable(cls, a, b) -> bool:
        """In the places where both PSs have fixed params, the values are the same"""
        for v_a, v_b in zip(a.values, b.values):
            if (v_a >= 0 and v_b >= 0) and (v_a != v_b):
                return False
        return True


    @classmethod
    def merge(cls, a, b):
        """assumes that mergeable(a, b) == True"""
        new_values = np.max((a, b), axis=0)
        return cls(new_values)
