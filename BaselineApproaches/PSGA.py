import random

from Evaluator import PSEvaluator
from GA import GA
from PS import PS, STAR
from SearchSpace import SearchSpace


class PSGA(GA):
    search_space: SearchSpace

    def __init__(self, search_space: SearchSpace,
                 mutation_rate: float,
                 crossover_rate: float,
                 elite_size: int,
                 tournament_size: int,
                 population_size: int,
                 ps_evaluator: PSEvaluator,
                 starting_population=None):

        super().__init__(mutation_rate=mutation_rate,
                         crossover_rate=crossover_rate,
                         elite_size=elite_size,
                         tournament_size=tournament_size,
                         population_size=population_size,
                         evaluator=ps_evaluator,
                         starting_population=starting_population)
        self.search_space = search_space

    def random_individual(self) -> PS:
        return PS.random(self.search_space, half_chance_star=True)

    def mutated(self, individual: PS) -> PS:
        result_values = individual.values.copy()

        for variable_index, cardinality in enumerate(self.search_space.cardinalities):
            if self.should_mutate():
                if random.random() < 0.5:
                    result_values[variable_index] = STAR
                else:
                    result_values[variable_index] = random.randrange(cardinality)
        return PS(result_values)

    def crossed(self, mother: PS, father: PS) -> PS:
        last_index = len(mother)
        start_cut = random.randrange(last_index)
        end_cut = random.randrange(last_index)
        start_cut, end_cut = min(start_cut, end_cut), max(start_cut, end_cut)

        def take_from(donor: PS, start_index, end_index) -> list[int]:
            return list(donor.values[start_index:end_index])

        child_value_list = (take_from(mother, 0, start_cut) +
                            take_from(father, start_cut, end_cut) +
                            take_from(mother, end_cut, last_index))

        return PS(child_value_list)
