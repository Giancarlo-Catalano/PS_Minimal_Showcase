import random
from typing import Iterable, Sized
import numpy as np

from custom_types import ArrayOfInts

from FullSolution import FullSolution


class SearchSpace(Sized):
    cardinalities: ArrayOfInts
    precomputed_offsets: ArrayOfInts

    def __init__(self, cardinalities: Iterable[int]):
        self.cardinalities = np.fromiter(cardinalities, dtype=int)
        self.precomputed_offsets = np.concatenate(([0], np.cumsum(self.cardinalities)))

    @property
    def hot_encoded_length(self) -> int:
        return np.sum(self.cardinalities)

    @property
    def dimensions(self) -> int:
        return len(self.cardinalities)

    def __len__(self) -> int:
        return self.dimensions

    @property
    def amount_of_parameters(self) -> int:
        return self.dimensions

    def get_random_fs(self) -> FullSolution:
        return FullSolution(random.randrange(card) for card in self.cardinalities)

    def __repr__(self):
        return f"SearchSpace{tuple(self.cardinalities)}"


    def __eq__(self, other) -> bool:
        return all(self.cardinalities == other.cardinalities)

    @classmethod
    def concatenate_search_spaces(cls, to_concat: Iterable):
        cardinalities: tuple[ArrayOfInts] = tuple(ss.cardinalities for ss in to_concat)
        return cls(np.concatenate(cardinalities))
