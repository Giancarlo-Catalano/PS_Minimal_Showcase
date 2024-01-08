import random
from typing import Iterable, Self
import numpy as np

from types import ParameterValue, ArrayOfInts

from FullSolution import FullSolution


class SearchSpace:
    cardinalities: ArrayOfInts
    precomputed_offsets: ArrayOfInts

    def __init__(self, cardinalities: Iterable[int]):
        self.cardinalities = np.array(cardinalities)
        self.precomputed_offsets = np.cumsum(self.cardinalities)

    @property
    def hot_encoded_length(self) -> int:
        return np.sum(self.cardinalities)

    @property
    def dimensions(self) -> int:
        return len(self.cardinalities)

    @property
    def amount_of_parameters(self) -> int:
        return self.dimensions

    def get_random_fs(self) -> FullSolution:
        return FullSolution(random.randrange(card) for card in self.cardinalities)

    def __repr__(self):
        return f"SearchSpace{tuple(self.cardinalities)}"

    @classmethod
    def concatenate_search_spaces(cls, to_concat: Iterable[Self]) -> Self:
        cardinalities: tuple[ArrayOfInts] = tuple(ss.cardinalities for ss in to_concat)
        return cls(np.concatenate(cardinalities))
