import random

from jmetal.core.operator import Mutation
from jmetal.core.solution import (
    IntegerSolution,
)
from jmetal.util.ckecking import Check

from PS import STAR, PS
from PSMiners.Operators.PSMutationOperator import MultimodalMutationOperator


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


class BidirectionalMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float):
        super(BidirectionalMutation, self).__init__(probability=probability)
        super().__init__(probability)

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        Check.that(issubclass(type(solution), IntegerSolution), "Solution type invalid")

        for i in range(solution.number_of_variables):
            if random.random() <= self.probability:
                ### 50% chance of becoming unfixed or fixed
                if random.random() < 0.5:
                    solution.variables[i] = random.randrange(solution.lower_bound[i], solution.upper_bound[i])
                else:
                    solution.variables[i] = STAR
        return solution

    def get_name(self):
        return "Bidirectional Mutation"


class HalfNHalfMutation(Mutation[IntegerSolution]):
    ps_mutation_operator: MultimodalMutationOperator

    def __init__(self, probability: float):
        super(HalfNHalfMutation, self).__init__(probability=probability)
        super().__init__(probability)
        self.ps_mutation_operator = MultimodalMutationOperator(probability)

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        Check.that(issubclass(type(solution), IntegerSolution), "Solution type invalid")

        as_ps = PS(solution.variables)
        mutated = self.ps_mutation_operator.mutated(as_ps)
        solution.variables = list(mutated.values)
        return solution

    def get_name(self):
        return "HalfNHalf Mutation"