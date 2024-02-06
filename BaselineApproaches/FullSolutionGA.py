import random
from typing import Callable

from GA import GA
from FullSolution import FullSolution
from Evaluator import Evaluator, Individual, FullSolutionEvaluator
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria
from custom_types import Fitness



class FullSolutionGA(GA):
    search_space: SearchSpace

    def __init__(self, search_space: SearchSpace,
                 mutation_rate: float,
                 crossover_rate: float,
                 elite_size: int,
                 tournament_size: int,
                 population_size: int,
                 termination_criteria: TerminationCriteria,
                 fitness_function: Callable,
                 starting_population=None):
        super().__init__(mutation_rate = mutation_rate,
                         crossover_rate=crossover_rate,
                         elite_size=elite_size,
                         tournament_size=tournament_size,
                         population_size=population_size,
                         termination_criteria=termination_criteria,
                         evaluator=FullSolutionEvaluator(fitness_function),
                         starting_population=starting_population)

        self.search_space = search_space


    def random_individual(self) -> FullSolution:
        return FullSolution.random(self.search_space)
    def mutated(self, individual: FullSolution) -> Individual:
        result = FullSolution(individual.values)
        for variable_index, cardinality in enumerate(self.search_space):
            if self.should_mutate():
                result.values[variable_index] = random.randrange(cardinality)
        return result


    def crossed(self, mother: FullSolution, father: FullSolution) -> FullSolution:
        last_index = len(mother)
        start_cut = random.randrange(last_index)
        end_cut = random.randrange(last_index)
        start_cut, end_cut = min(start_cut, end_cut), max(start_cut, end_cut)

        def take_from(donor: FullSolution, start_index, end_index) -> list[int]:
            return list(donor.values[start_index:end_index])

        child_value_list = (take_from(mother, 0, start_cut) +
                            take_from(father, start_cut, end_cut) +
                            take_from(mother, end_cut, last_index))

        return FullSolution(child_value_list)


