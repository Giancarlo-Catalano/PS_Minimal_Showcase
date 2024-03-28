import heapq
import random
from typing import Callable

import utils
from BaselineApproaches import Selection
from BaselineApproaches.Evaluator import Evaluator, Individual, Population, EvaluatedPopulation, EvaluatedIndividual
from BaselineApproaches.Selection import tournament_select
from EDA.FSEvaluator import FSEvaluator
from EDA.FSIndividual import FSIndividual
from FullSolution import FullSolution
from TerminationCriteria import TerminationCriteria


class GA:
    mutation_rate: float
    crossover_rate: float

    elite_size: int
    tournament_size: int
    population_size: int
    termination_criteria: TerminationCriteria

    evaluator: FSEvaluator

    current_population: list[FSIndividual]

    def __init__(self,
                 mutation_rate: float,
                 crossover_rate: float,
                 elite_size: int,
                 tournament_size: int,
                 population_size: int,
                 fitness_function: Callable[[FullSolution], float],
                 starting_population=None):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.population_size = population_size
        self.evaluator = FSEvaluator(fitness_function)

        if starting_population is None:
            self.current_population = self.get_initial_population()
        else:
            self.current_population = starting_population

        self.current_population = self.evaluator.evaluate_population(self.current_population)

    def random_solution(self) -> FullSolution:
        raise Exception("An implementation of GA.random_individual is not valid")

    def get_initial_population(self) -> list[FSIndividual]:
        return [FSIndividual(self.random_solution(), 0) for _ in range(self.population_size)]

    def should_mutate(self) -> bool:
        return random.random() < self.mutation_rate

    def mutated(self, individual: FullSolution) -> FullSolution:
        raise Exception("An implementation of GA.mutated is not valid")

    def crossed(self, mother: FullSolution, father: FullSolution) -> FullSolution:
        raise Exception("An implementation of GA.crossed is not valid")

    def select(self) -> FSIndividual:
        """tournament selection"""
        return max(random.choices(self.current_population, k=self.tournament_size))

    def get_elite(self) -> list[FSIndividual]:
        return heapq.nlargest(self.elite_size, self.current_population)

    def make_new_child(self) -> FSIndividual:
        if random.random() < self.crossover_rate:
            # do crossover
            mother = self.select().full_solution
            father = self.select().full_solution

            child_ps = self.mutated(self.crossed(mother, father))
        else:
            child_ps = self.mutated(self.select().full_solution)
        return FSIndividual(child_ps, 0)

    def make_new_evaluated_population(self) -> list[FSIndividual]:
        elite = self.get_elite()
        children = [self.make_new_child()
                    for _ in range(self.population_size - self.elite_size)]
        children = self.evaluator.evaluate_population(children)
        return elite + children

    def step(self):
        self.current_population = self.make_new_evaluated_population()

    def run(self,
            termination_criteria: TerminationCriteria,
            show_every_generation=False):
        iteration = 0

        def termination_criteria_met():
            return termination_criteria.met(iterations=iteration,
                                            evaluations=self.evaluator.used_evaluations,
                                            evaluated_population=self.current_population)

        while not termination_criteria_met():
            if show_every_generation:
                self.show_current_state()
            self.step()
            iteration += 1

    def get_current_best(self) -> FSIndividual:
        return max(self.current_population)

    def show_current_state(self):
        print(f"The current best fitness is {self.get_current_best()}.")
