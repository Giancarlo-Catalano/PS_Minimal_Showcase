import random

from jmetal.core.operator import Mutation
from jmetal.core.solution import (
    BinarySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
    PermutationSolution,
    Solution,
)
from jmetal.util.ckecking import Check

from PS import STAR





class SpecialisationMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float):
        super(SpecialisationMutation, self).__init__(probability=probability)
        super().__init__(probability)

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        Check.that(issubclass(type(solution), IntegerSolution), "Solution type invalid")

        for i in range(solution.number_of_variables):
            if solution.variables[i] == STAR and random.random() <= self.probability:
                solution.variables[i] = random.randrange(solution.lower_bound[i], solution.upper_bound[i])
        return solution

    def get_name(self):
        return "Specialisation Mutation"
