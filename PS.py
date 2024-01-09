from typing import Iterable, Self, Sized

import numpy as np
from bitarray import bitarray, frozenbitarray

from FullSolution import FullSolution
from SearchSpace import SearchSpace
from custom_types import ArrayOfInts


class PS(Sized):
    fixed_mask: frozenbitarray
    values: ArrayOfInts

    def __init__(self, fixed_mask: bitarray | frozenbitarray, values: Iterable[int]):
        self.fixed_mask = frozenbitarray(fixed_mask)
        self.values = np.fromiter(values, dtype=int)

    def __len__(self):
        return len(self.fixed_mask)

    def __repr__(self) -> str:
        def repr_single(cell_value: int, is_fixed: bool) -> str:
            return f'{cell_value}' if is_fixed else '*'

        return "[" + "".join(map(repr_single, self.fixed_mask, self.values))

    @classmethod
    def empty(cls, search_space: SearchSpace) -> Self:
        no_fixed_values = bitarray(len(search_space))
        no_fixed_values.setall(False)

        irrelevant_values = np.zeros(len(search_space), dtype=int)

        return cls(no_fixed_values, irrelevant_values)

    def is_fully_fixed(self) -> bool:
        return self.fixed_mask.all()

    def to_FS(self) -> FullSolution:
        return FullSolution(self.values)

    @classmethod
    def from_FS(cls, fs: FullSolution) -> Self:
        all_fixed = bitarray(len(fs))
        all_fixed.setall(True)
        return cls(all_fixed, fs.values)

    def with_unfixed_value(self, variable_position: int) -> Self:
        new_fixed_mask = bitarray(self.fixed_mask)
        new_fixed_mask[variable_position] = False
        return PS(new_fixed_mask, self.values)

    def with_fixed_value(self, variable_position: int, fixed_value: int) -> Self:
        new_fixed_mask = bitarray(self.fixed_mask)
        new_fixed_mask[variable_position] = True
        new_values = np.array(self.values)
        new_values[variable_position] = fixed_value
        return PS(new_fixed_mask, new_values)

    def get_fixed_variable_positions(self) -> list[int]:
        return [position for position, is_used in enumerate(self.fixed_mask) if is_used]

    def get_unfixed_variable_positions(self) -> list[int]:
        return [position for position, is_used in enumerate(self.fixed_mask) if not is_used]

    def simplifications(self) -> list[Self]:
        return [self.with_unfixed_value(i) for i in self.get_fixed_variable_positions()]

    def specialisations(self, search_space: SearchSpace) -> list[Self]:
        return [self.with_fixed_value(position, value)
                for position in self.get_unfixed_variable_positions()
                for value in range(search_space.cardinalities[position])]

    def __eq__(self, other: Self) -> bool:
        """ This had to be optimised"""
        if self.fixed_mask != other.fixed_mask:
            return False

        # shortcircuit on completely set features
        if self.fixed_mask.all():
            return np.array_equal(self.values, other.values_mask)

        # full equality check
        for value_here, value_there, is_set in zip(self.values, other.values_mask, self.fixed_mask):
            if is_set and value_here != value_there:
                return False

        return True


    @classmethod
    def mergeable(cls, a: Self, b: Self) -> bool:
        """In the places where both PSs have fixed params, the values are the same"""
        for v_a, v_b, f_a, f_b in zip(a.values, b.values, a.fixed_mask, b.fixed_mask):
            if (f_a and f_b) and (v_a != v_b):
                return False
        return True


    @classmethod
    def merge(cls, a: Self, b: Self) -> Self:
        """assumes that mergeable(a, b) == True"""
        merged_fixed_mask = a.fixed_mask | b.fixed_mask
        merged_values = np.array(a.values)
        for fixed_par_in_b in b.get_fixed_variable_positions():
            merged_values[fixed_par_in_b] = b.values[fixed_par_in_b]

        return cls(merged_fixed_mask, merged_values)
