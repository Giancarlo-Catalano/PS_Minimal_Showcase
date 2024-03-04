import random
from typing import Optional

import numpy as np

from PS import PS, STAR
from SearchSpace import SearchSpace


class PSMutationOperator:
    search_space: Optional[SearchSpace]

    def __init__(self):
        self.search_space = None

    def set_search_space(self, search_space: SearchSpace):
        self.search_space = search_space

    def __repr__(self):
        return "PSMutationOperator"


    def mutated(self, ps: PS) -> PS:
        raise Exception("An implementation of PSMutationOperator does not implement .mutated")


class SinglePointMutation(PSMutationOperator):
    mutation_probability: float
    chance_of_unfixing: float
    def __init__(self,
                 probability: float,
                 chance_of_unfixing: float):
        self.mutation_probability = probability
        self.chance_of_unfixing = chance_of_unfixing
        super().__init__()


    def __repr__(self):
        return "SinglePointMutation"


    def mutated(self, ps: PS) -> PS:
        def get_mutated_value_for(index: int):
            cardinality = self.search_space.cardinalities[index]

            if ps.values[index] == STAR:
                return random.randrange(cardinality)
            else:
                if random.random() < self.chance_of_unfixing:
                    return STAR
                else:
                    return random.randrange(cardinality)

        new_values = np.copy(ps.values)
        for index in range(len(new_values)):
            if random.random() < self.mutation_probability:
                new_values[index] = get_mutated_value_for(index)

        return PS(new_values)
