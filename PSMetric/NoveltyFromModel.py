from typing import TypeAlias, Optional

import numpy as np

from PRef import PRef
from PS import PS
from PSMetric.Metric import Metric
from PSMiners.Individual import Individual
from SearchSpace import SearchSpace
from custom_types import ArrayOfFloats

Model: TypeAlias = list[Individual]

class NoveltyFromModel(Metric):
    reference_model: Optional[Model]
    frequencies: Optional[list[ArrayOfFloats]]
    search_space: Optional[SearchSpace]

    def __init__(self):
        self.reference_model = None
        self.frequencies = None
        super().__init__()


    def __repr__(self):
        return "NoveltyFromModel"


    def set_pRef(self, pRef: PRef):
        self.search_space = pRef.search_space


    @staticmethod
    def generate_frequencies_from_model(model: Model, search_space: SearchSpace) -> list[ArrayOfFloats]:
        # the first slot is for *, the second is for 0, the third is for 1 etc..
        counts = [np.array([0.0 for value in range(cardinality+1)])
                       for cardinality in search_space.cardinalities]

        for item in model:
            for variable, value in enumerate(item.ps.values):
                counts[variable][value+1] += 1.0  # note that we're adding one to convert * -> 0, 0 -> 1 etc

        model_size = len(model)
        frequencies = [counts_for_var / model_size for counts_for_var in counts]
        return frequencies

    def set_reference_model(self, model: Model):
        self.reference_model = model
        self.frequencies = self.generate_frequencies_from_model(model, self.search_space)


    def get_single_normalised_score(self, ps: PS) -> float:
        frequencies_of_present_values = [self.frequencies[var][val+1] for var, val in enumerate(ps.values)]
        return 1-np.average(frequencies_of_present_values)
