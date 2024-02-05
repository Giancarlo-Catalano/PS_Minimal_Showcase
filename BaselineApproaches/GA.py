import heapq
import random
from typing import Callable, TypeAlias, Any

import utils
from BaselineApproaches.Evaluator import Evaluator, Individual, Population, EvaluatedPopulation, EvaluatedIndividual
from TerminationCriteria import TerminationCriteria


class GA:
    mutation_rate: float
    crossover_rate: float

    elite_size: int
    tournament_size: int
    population_size: int
    termination_criteria: TerminationCriteria

    evaluator: Evaluator

    last_evaluated_population: EvaluatedPopulation

    def __init__(self,
                 mutation_rate: float,
                 crossover_rate: float,
                 elite_size: int,
                 tournament_size: int,
                 population_size: int,
                 termination_criteria: TerminationCriteria,
                 fitness_function: Callable,
                 starting_population=None):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.termination_criteria = termination_criteria
        self.evaluator = Evaluator(fitness_function)

        if starting_population is None:
            self.last_evaluated_population = self.get_initial_population()
        else:
            self.last_evaluated_population = starting_population

    def random_individual(self) -> Individual:
        raise Exception("An implementation of GA.random_individual is not valid")

    def get_initial_population(self) -> Population:
        return [self.random_individual() for _ in range(self.population_size)]

    def mutated(self, individual: Individual) -> Individual:
        raise Exception("An implementation of GA.mutated is not valid")

    def crossed(self, mother: Individual, father: Individual) -> Individual:
        raise Exception("An implementation of GA.crossed is not valid")

    def without_fitnesses(self, evaluated_population: EvaluatedPopulation) -> Population:
        if len(evaluated_population) == 0:
            return []
        return utils.unzip(evaluated_population)[0]

    def tournament_select(self) -> Individual:
        tournament_pool = random.choices(self.last_evaluated_population, k=self.tournament_size)
        winner = max(tournament_pool, key=utils.second)[0]
        return winner

    def get_elite(self) -> Population:
        top_evaluated = heapq.nlargest(self.elite_size, self.last_evaluated_population, key=utils.second)
        if len(top_evaluated) == 0:
            return []
        else:
            return self.without_fitnesses(top_evaluated)

    def make_new_child(self) -> Individual:
        if random.random() < self.crossover_rate:
            # do crossover
            mother = self.tournament_select()
            father = self.tournament_select()

            return self.mutated(self.crossed(mother, father))
        else:
            return self.mutated(self.tournament_select())

    def make_new_evaluated_population(self, evaluated_population: EvaluatedPopulation) -> EvaluatedPopulation:
        elite = self.get_elite()
        children = [self.make_new_child()
                    for _ in range(self.population_size - self.elite_size)]
        return self.evaluator.evaluate_population(elite + children)

    def evolve_for_generations(self):
        self.last_evaluated_population = self.evaluator.evaluate_population(self.last_evaluated_population)
        iteration = 0

        def termination_criteria_met():
            return self.termination_criteria.met(iteration=iteration,
                                                 evaluations=self.evaluator.used_evaluations,
                                                 evaluated_population=self.last_evaluated_population)

        while not termination_criteria_met():
            self.last_evaluated_population = self.make_new_evaluated_population(self.last_evaluated_population)

    def get_current_best(self) -> EvaluatedIndividual:
        return max(self.last_evaluated_population, key=utils.second)
