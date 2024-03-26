import heapq
import random
import warnings
from math import ceil
from typing import TypeAlias

from pandas import DataFrame

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import MultipleMetrics
from PSMetric.Simplicity import Simplicity
from PSMiners.Individual import Individual, add_metrics, with_aggregated_scores, add_normalised_metrics, \
    with_average_score, with_product_score, partition_by_simplicity
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria, EvaluationBudgetLimit, AsLongAsWanted
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

Population: TypeAlias = set[Individual]
Archive: TypeAlias = set[Individual]

class EfficientArchiveMiner:
    population_size: int
    metrics: MultipleMetrics

    current_population: Population
    archive: set[Individual]

    selection_proportion = 0.3
    tournament_size = 3

    search_space: SearchSpace

    def __init__(self,
                 population_size: int,
                 pRef: PRef,
                 metrics=None):
        self.search_space = pRef.search_space
        self.population_size = population_size
        if metrics is None:
            self.metrics = MultipleMetrics([Simplicity(), MeanFitness(), Linkage()])
        else:
            self.metrics = metrics
        self.metrics.set_pRef(pRef)

        print(f"Initialised the ArchiveMiner with metrics {self.metrics.get_labels()}")

        self.current_population = self.evaluate(self.make_initial_population())
        self.archive = set()

    def evaluate(self, population: Population) -> Population:
        for individual in population:
            scores = self.metrics.get_normalised_scores(individual.ps)
            individual.metric_scores = scores
            individual.aggregated_score = scores[0] * scores[1]
        return population


    def current_best_atomicity(self) -> float:
        return max(self.current_population, key=lambda x: x.metric_scores[1]).metric_scores[1]

    def make_initial_population(self) -> Population:
        """ basically takes the elite of the PRef, and converts them into PSs """
        """this is called get_init in the paper"""
        return {Individual(PS.empty(self.search_space))}

    def get_localities(self, individual: Individual) -> list[Individual]:
        return [Individual(ps) for ps in individual.ps.specialisations(self.search_space)]

    def select_one(self) -> Individual:
        tournament_pool = random.choices(self.current_population, k=self.tournament_size)  # TODO
        winning_individual = max(tournament_pool, key=lambda x: x.aggregated_score)
        return winning_individual

    @staticmethod
    def top(evaluated_population: Population, amount: int) -> list[Individual]:
        return heapq.nlargest(amount, evaluated_population, key=lambda x: x.aggregated_score)


    def select_from_current_population(self) -> list[Individual]:
        population_as_list = list(self.current_population)

        def select_one():
            tournament_pool = random.choices(population_as_list, k=self.tournament_size)
            winning_individual = max(tournament_pool, key=lambda x: x.aggregated_score)
            return winning_individual

        amount_to_select = ceil(self.population_size * self.selection_proportion)
        return [select_one() for _ in range(amount_to_select)]

    def update_population(self):
        """Note how this relies on the current population being evaluated,
                and how at the end it returns an evaluated population"""
        # select and add to archive, so that they won't appear in the population again
        selected_individuals = self.select_from_current_population()
        self.archive.update(selected_individuals)

        # get the neighbourhoods of those selected individuals, add them to the population
        localities = {local for selected_single in selected_individuals
                            for local in self.get_localities(selected_single)}
        localities = self.evaluate(localities)

        self.current_population.update(localities)

        # remove unwanted individuals from the population
        self.current_population.difference_update(self.archive)

        self.current_population = set(self.top(self.current_population, self.population_size))

    def show_best_of_current_population(self, how_many: int):
        best = self.top(self.current_population, how_many)
        for individual in best:
            print(individual)

    def run(self,
            termination_criteria: TerminationCriteria,
            show_each_generation=False):
        iteration = 0

        def termination_criteria_met():
            if len(self.current_population) == 0:
                warnings.warn("The run is ending because the population is empty!!!")
                return True

            """ This is the new rule: if the atomicity dips below 0.5, it means that the population is irremediably bad"""
            if iteration > 3 and self.current_best_atomicity() < 0.5:
                print("Terminating because the current best atomicity is less than 0.5")
                return True

            return termination_criteria.met(iterations=iteration,
                                            evaluations=self.get_used_evaluations(),
                                            evaluated_population=self.current_population)  # TODO change this

        while not termination_criteria_met():
            self.update_population()
            if show_each_generation:
                print(f"Population at iteration {iteration}, used_budget = {self.get_used_evaluations()}--------------")
                self.show_best_of_current_population(12)
            iteration += 1

        print(f"Execution terminated with {iteration = } and used_budget = {self.get_used_evaluations()}")

    def get_results(self, quantity_returned=None) -> list[Individual]:
        if quantity_returned is None:
            quantity_returned = len(self.archive)
        return list(self.top(self.archive, quantity_returned))

    def get_used_evaluations(self) -> int:
        return self.metrics.used_evaluations


def show_plot_of_individuals(individuals: list[Individual], metrics: MultipleMetrics):
    labels = metrics.get_labels()
    points = [i.metric_scores for i in individuals]

    utils.make_interactive_3d_plot(points, labels)


def test_efficient_archive_miner(problem: BenchmarkProblem,
                       show_each_generation=True,
                       show_interactive_plot=False,
                       metrics=None):
    print(f"Testing the modified archive miner")
    pRef: PRef = problem.get_pRef(15000)

    budget_limit = AsLongAsWanted()
    # iteration_limit = TerminationCriteria.IterationLimit(12)
    # termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, iteration_limit)

    miner = EfficientArchiveMiner(150,
                                  pRef,
                                  metrics=metrics)

    miner.run(budget_limit, show_each_generation=show_each_generation)

    results = miner.get_results()
    # print("The results of the archive miner are:")
    # for individual in results:
    #     print(f"{individual}")

    print(f"The used budget is {miner.get_used_evaluations()}")

    if show_interactive_plot:
        print("Displaying the plot")
        show_plot_of_individuals(results, miner.metrics)

    # print("Partitioned by complexity, the PSs are")
    # for fixed_count, pss in enumerate(partition_by_simplicity(results)):
    #     if len(pss) > 0:
    #         print(f"With {fixed_count} fixed vars: --------")
    #         for individual in pss:
    #             print(individual)

    print("The top 12 by mean fitness are")
    sorted_by_mean_fitness = sorted(results, key=lambda i: i.aggregated_score, reverse=True)
    for individual in sorted_by_mean_fitness[:12]:
        print(individual)
