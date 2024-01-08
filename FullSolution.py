from typing import Iterable
import numpy as np
from types import ParameterValue, ArrayOfInts


class FullSolution:
    # a wrapper for a tuple
    values: ArrayOfInts

    def __init__(self, values: Iterable[ParameterValue]):
        self.values = np.array(values)
        self.values.setflags(write=False)

    def __repr__(self):
        return "(" + (" ".join([f"{val}" for val in self.values])) + ")"

    def __eq__(self, other):
        return self.values == other.values

    def __hash__(self):
        return self.values.__hash__()
