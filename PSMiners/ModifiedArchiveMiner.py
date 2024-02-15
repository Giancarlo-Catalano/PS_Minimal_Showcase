import heapq
import random
import warnings
from math import ceil
from typing import TypeAlias

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.KindaAtomicity import SimplerAtomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import ManyMetrics
from PSMetric.Simplicity import Simplicity
from PSMiners.Individual import Individual, add_metrics, with_aggregated_scores
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria
from custom_types import ArrayOfFloats


class ModifiedArchiveMiner:
    population_size: int
    pRef: PRef
    many_metrics: ManyMetrics

    current_population: list[Individual]
    archive: set[PS]

    selection_proportion = 0.3
    tournament_size = 3

    def __init__(self,
                 population_size: int,
                 pRef: PRef):
        self.population_size = population_size
        self.pRef = pRef
        self.many_metrics = ManyMetrics([Simplicity(), MeanFitness(), Atomicity()])

        self.current_population = self.calculate_metrics_and_aggregated_score(self.make_initial_population())
        self.archive = set()


    def calculate_metrics_and_aggregated_score(self, population: list[Individual]):
        to_return = add_metrics(population, self.many_metrics)
        return with_aggregated_scores(to_return)


    @property
    def search_space(self) -> SearchSpace:
        return self.pRef.search_space

    def make_initial_population(self) -> list[Individual]:
        """ basically takes the elite of the PRef, and converts them into PSs """
        """this is called get_init in the paper"""
        return [Individual(PS.empty(self.search_space))]

    def get_localities(self, ps: PS) -> list[PS]:
        return ps.specialisations(self.search_space)

    def select_one(self) -> PS:
        tournament_pool = random.choices(self.current_population, k=self.tournament_size)
        winning_individual = max(tournament_pool, key=lambda x: x.aggregated_score)
        return winning_individual.ps

    @staticmethod
    def top(evaluated_population: list[Individual], amount: int) -> list[Individual]:
        return heapq.nlargest(amount, evaluated_population, key=lambda x: x.aggregated_score)

    def make_selection(self) -> list[PS]:
        amount_of_select = ceil(self.population_size * self.selection_proportion)
        if len(self.current_population) < amount_of_select:
            return [individual.ps for individual in self.current_population]
        else:
            return [self.select_one() for _ in range(amount_of_select)]

    def make_new_evaluated_population(self):
        selected = [self.select_one() for _ in range(ceil(self.population_size * self.selection_proportion))]
        localities = [local for ps in selected
                      for local in self.get_localities(ps)]
        self.archive.update(selected)

        # make a population with the current + localities
        new_population = [i.ps for i in self.current_population] + localities

        # remove selected individuals and those in the archive
        new_population = [ps for ps in new_population if ps not in self.archive]

        # remove duplicates
        new_population = list(set(new_population))

        # convert to individual and evaluate
        new_population = [Individual(ps) for ps in new_population]
        new_population = self.calculate_metrics_and_aggregated_score(new_population)

        return self.top(new_population, self.population_size)

    def run(self, termination_criteria: TerminationCriteria):
        iteration = 0

        def termination_criteria_met():
            if len(self.current_population) == 0:
                warnings.warn("The run is ending because the population is empty!!!")
                return True

            return termination_criteria.met(iterations=iteration,
                                            evaluations=self.get_used_evaluations(),
                                            evaluated_population=self.current_population)  # TODO change this

        while not termination_criteria_met():
            self.current_population = self.make_new_evaluated_population()
            iteration += 1

        print(f"Execution terminated with {iteration = } and used_budget = {self.get_used_evaluations()}")

    def get_results(self, quantity_returned: int) -> list[(PS, float)]:
        evaluated_archive = self.calculate_metrics_and_aggregated_score([Individual(ps) for ps in self.archive])
        best = self.top(evaluated_archive, quantity_returned)
        return [(i.ps, i.aggregated_score) for i in best]

    def get_used_evaluations(self) -> int:
        return self.many_metrics.used_evaluations



def test_modified_archive_miner(problem: BenchmarkProblem):
    print("Testing the modified archive miner")
    pRef: PRef = problem.get_pRef(10000)

    budget_limit = TerminationCriteria.EvaluationBudgetLimit(10000)
    # iteration_limit = TerminationCriteria.IterationLimit(12)
    # termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, iteration_limit)

    miner = ModifiedArchiveMiner(150, pRef)

    miner.run(budget_limit)

    results = miner.get_results(quantity_returned=10)
    print("The results of the PSABSM are:")
    for ps, fitness in results:
        print(f"PS: {ps}, fitness = {fitness}")

    print(f"The used budget is {miner.get_used_evaluations()}")
