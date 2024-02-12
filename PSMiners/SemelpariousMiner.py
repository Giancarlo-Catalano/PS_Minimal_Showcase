import heapq
import random
import warnings
from math import ceil
from typing import TypeAlias, Callable

import utils
from BaselineApproaches import Selection
from BaselineApproaches.Evaluator import PSEvaluator
from JMetal.PSProblem import AtomicityEvaluator
from PRef import PRef
from PS import PS
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria
from BaselineApproaches.Selection import tournament_select, tournament_generic

Metrics: TypeAlias = tuple


class Individual:
    ps: PS
    has_reproduced: bool

    # TODO make this more flexible! What if there were more metrics?
    metrics: list  # simplicity, meanFitness, atomicity

    def __init__(self, ps: PS):
        self.ps = ps
        self.has_reproduced = False

        # TODO find metter default values...
        self.metrics = [None, None, None]

    def get_metric(self, metric_index: int) -> float:
        return self.metrics[metric_index]

    def __repr__(self):
        return f"<{self.ps}, {self.has_reproduced}, {self.metrics}>"

    def __hash__(self):
        return hash(tuple(self.metrics))

    def __eq__(self, other):
        return self.ps == other.ps


class SemelpariousMiner:
    population_size: int
    pRef: PRef
    atomicity_evaluator: AtomicityEvaluator
    simplicity: Simplicity()
    mean_fitness: MeanFitness()

    current_population: list[Individual]
    available_from_current_population: int

    evaluations: int

    tournament_size = 3
    selection_proportion = 0.3

    def __init__(self,
                 population_size: int,
                 pRef: PRef):
        self.population_size = population_size
        self.pRef = pRef
        self.atomicity_evaluator = AtomicityEvaluator(self.pRef)
        self.simplicity = Simplicity()
        self.mean_fitness = MeanFitness()
        self.evaluations = 0
        self.current_population = self.evaluate_population(self.make_initial_population())

        self.available_from_current_population = 1  # dummy value
        self.update_available_count()

    @property
    def search_space(self) -> SearchSpace:
        return self.pRef.search_space

    def make_initial_population(self) -> list[Individual]:
        """this is called get_init in the paper"""
        return [Individual(PS.empty(self.search_space))]

    def evaluate_individual(self, individual: Individual) -> Individual:
        # sets the values in the individual and returns it, similarly to how JMetal does it...
        individual.metrics[0] = self.mean_fitness.get_single_unnormalised_score(individual.ps, self.pRef)
        individual.metrics[1] = self.simplicity.get_single_unnormalised_score(individual.ps, self.pRef)
        individual.metrics[2] = self.atomicity_evaluator.evaluate_single(individual.ps)

        self.evaluations += 1
        return individual

    def evaluate_population(self, population: list[Individual]) -> list[Individual]:
        return [self.evaluate_individual(individual) for individual in population]

    def get_localities(self, individual: Individual) -> list[PS]:
        # as seen in the paper
        return individual.ps.specialisations(self.search_space)

    def select_from_population_unsafe(self) -> Individual:
        random_key = SemelpariousMiner.get_key_metric()
        tournament_pool = random.choices(self.current_population, k=self.tournament_size)
        winner = max(tournament_pool, key=random_key)
        return winner

    def select_from_population(self) -> Individual:
        current_winner = self.select_from_population_unsafe()

        attempts = 0
        while current_winner.has_reproduced:
            current_winner = self.select_from_population_unsafe()
            attempts += 1

        return current_winner

    def get_offspring(self, individual: Individual) -> list[Individual]:
        localities_of_ps = individual.ps.specialisations(self.search_space)
        localities = [Individual(ps) for ps in localities_of_ps]
        localities = self.evaluate_population(localities)
        return localities

    @staticmethod
    def get_key_metric() -> Callable:
        metric_index = random.randrange(3)  # Hardcoded, fix this

        def from_chosen_metric(individual: Individual) -> float:
            return individual.metrics[metric_index]

        return from_chosen_metric

    @staticmethod
    def top(population: list[Individual], amount: int) -> list[Individual]:
        return heapq.nlargest(amount, population, key=SemelpariousMiner.get_key_metric())
    def make_new_population(self) -> list[Individual]:
        print(f"Currently, {self.available_from_current_population} individuals can be selected")

        amount_to_select = ceil(len(self.current_population) / 3)
        offspring = []
        for _ in range(amount_to_select):
            if self.get_percentage_of_selectable() < 0.10:
                break
            selected = self.select_from_population()
            locality = self.get_offspring(selected)
            selected.has_reproduced = True
            self.available_from_current_population -= 1
            offspring.extend(locality)

        self.current_population.extend(offspring)

        # top should work differently...
        return self.top(self.current_population, self.population_size)

    def remove_duplicates_from_population(self, population: list[Individual]) -> list[Individual]:
        return list(set(population))

    def update_available_count(self):
        self.available_from_current_population = len(
            [None for individual in self.current_population if not individual.has_reproduced])


    def get_percentage_of_selectable(self):
        return self.available_from_current_population / len(self.current_population)

    def run(self, termination_criteria: TerminationCriteria):
        iteration = 0

        def termination_criteria_met():
            if len(self.current_population) == 0:
                warnings.warn("The run is ending because the population is empty!!!")
                return True

            if self.get_percentage_of_selectable() < 0.1:
                warnings.warn(f"The run is ending because "
                              f"the there's not enough selectables ({self.available_from_current_population})!!!")

                return True

            return termination_criteria.met(iterations=iteration,
                                            evaluations=self.evaluations,
                                            evaluated_population=self.current_population)

        while not termination_criteria_met():
            self.current_population = self.make_new_population()
            self.current_population = self.remove_duplicates_from_population(self.current_population)
            self.update_available_count()
            iteration += 1

    def get_results(self, quantity_returned: int) -> list[(PS, float)]:
        return self.remove_duplicates_from_population(self.current_population)
