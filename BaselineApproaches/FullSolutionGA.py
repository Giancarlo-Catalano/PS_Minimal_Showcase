import random
from typing import Callable

from BaselineApproaches.Evaluator import Individual, FullSolutionEvaluator
from FullSolution import FullSolution
from BaselineApproaches.GA import GA
from SearchSpace import SearchSpace


class FullSolutionGA(GA):
    search_space: SearchSpace

    def __init__(self, search_space: SearchSpace,
                 mutation_rate: float,
                 crossover_rate: float,
                 elite_size: int,
                 tournament_size: int,
                 population_size: int,
                 fitness_function: Callable,
                 starting_population=None):


        self.search_space = search_space
        super().__init__(mutation_rate=mutation_rate,
                         crossover_rate=crossover_rate,
                         elite_size=elite_size,
                         tournament_size=tournament_size,
                         population_size=population_size,
                         evaluator=FullSolutionEvaluator(fitness_function),
                         starting_population=starting_population)



    def random_individual(self) -> FullSolution:
        return FullSolution.random(self.search_space)

    def mutated(self, individual: FullSolution) -> Individual:
        result_values = individual.values.copy()
        for variable_index, cardinality in enumerate(self.search_space.cardinalities):
            if self.should_mutate():
                result_values[variable_index] = random.randrange(cardinality)
        return FullSolution(result_values)

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
