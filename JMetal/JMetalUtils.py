from jmetal.core.solution import IntegerSolution

from PS import PS


def into_PS(metal_solution: IntegerSolution) -> PS:
    return PS(metal_solution.variables)