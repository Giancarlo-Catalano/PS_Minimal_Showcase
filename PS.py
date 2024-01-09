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

        return "["+"".join(map(repr_single, self.fixed_mask, self.values))


    def __len__(self):
        return len(self.fixed_mask)


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


