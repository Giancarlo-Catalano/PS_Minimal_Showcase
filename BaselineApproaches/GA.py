import heapq
import random

import utils
from BaselineApproaches.Evaluator import Evaluator, Individual, Population, EvaluatedPopulation, EvaluatedIndividual
from BaselineApproaches.Selection import tournament_select
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
                 evaluator: Evaluator,
                 starting_population=None):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.evaluator = evaluator

        if starting_population is None:
            self.last_evaluated_population = self.get_initial_population()
        else:
            self.last_evaluated_population = starting_population

    def random_individual(self) -> Individual:
        raise Exception("An implementation of GA.random_individual is not valid")

    def get_initial_population(self) -> Population:
        return [self.random_individual() for _ in range(self.population_size)]

    def should_mutate(self) -> bool:
        return random.random() < self.mutation_rate

    def mutated(self, individual: Individual) -> Individual:
        raise Exception("An implementation of GA.mutated is not valid")

    def crossed(self, mother: Individual, father: Individual) -> Individual:
        raise Exception("An implementation of GA.crossed is not valid")

    def without_fitnesses(self, evaluated_population: EvaluatedPopulation) -> Population:
        if len(evaluated_population) == 0:
            return []
        return utils.unzip(evaluated_population)[0]

    def select(self) -> Individual:
        return tournament_select(self.last_evaluated_population, self.tournament_size)

    def get_elite(self) -> Population:
        top_evaluated = heapq.nlargest(self.elite_size, self.last_evaluated_population, key=utils.second)
        if len(top_evaluated) == 0:
            return []
        else:
            return self.without_fitnesses(top_evaluated)

    def make_new_child(self) -> Individual:
        if random.random() < self.crossover_rate:
            # do crossover
            mother = self.select()
            father = self.select()

            return self.mutated(self.crossed(mother, father))
        else:
            return self.mutated(self.select())

    def make_new_evaluated_population(self) -> EvaluatedPopulation:
        elite = self.get_elite()
        children = [self.make_new_child()
                    for _ in range(self.population_size - self.elite_size)]
        return self.evaluator.evaluate_population(elite + children)

    def run(self, termination_criteria: TerminationCriteria):
        self.last_evaluated_population = self.evaluator.evaluate_population(self.last_evaluated_population)
        iteration = 0

        def termination_criteria_met():
            return termination_criteria.met(iteration=iteration,
                                            evaluations=self.evaluator.used_evaluations,
                                            evaluated_population=self.last_evaluated_population)

        while not termination_criteria_met():
            self.last_evaluated_population = self.make_new_evaluated_population()

    def get_current_best(self) -> EvaluatedIndividual:
        return max(self.last_evaluated_population, key=utils.second)
