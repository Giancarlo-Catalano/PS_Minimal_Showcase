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
from PSMiners.SamplableSet import EfficientPopulation
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria, EvaluationBudgetLimit, AsLongAsWanted
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# performance issues:
# eq, required by set

Population: TypeAlias = list[Individual]


class ThirdArchiveMiner:
    population_size: int
    metrics: MultipleMetrics

    current_population: EfficientPopulation

    selection_proportion = 0.3
    tournament_size = 3

    search_space: SearchSpace

    def __init__(self,
                 population_size: int,
                 pRef: PRef):
        self.search_space = pRef.search_space
        self.population_size = population_size
        self.metrics = MultipleMetrics([MeanFitness(), Linkage()])
        self.metrics.set_pRef(pRef)

        print(f"Initialised the ArchiveMiner with metrics {self.metrics.get_labels()}")

        self.current_population = EfficientPopulation(initial_population=self.evaluate(self.make_initial_population()),
                                                      max_size=self.population_size,
                                                      tournament_size=self.tournament_size)
        self.archive = set()

    def evaluate(self, population: Population) -> Population:
        for individual in population:
            scores = self.metrics.get_normalised_scores(individual.ps)
            individual.metric_scores = scores
            individual.aggregated_score = (scores[0] + scores[1]) / 2
        return population

    def current_best_atomicity(self) -> float:
        return max(self.current_population.current_population, key=lambda x: x.metric_scores[1]).metric_scores[1]

    def make_initial_population(self) -> Population:
        """ basically takes the elite of the PRef, and converts them into PSs """
        """this is called get_init in the paper"""
        return [Individual(PS.empty(self.search_space))]

    def get_localities(self, individual: Individual) -> list[Individual]:
        return [Individual(ps) for ps in individual.ps.specialisations(self.search_space)]

    def update_population(self):
        """Note how this relies on the current population being evaluated,
                and how at the end it returns an evaluated population"""
        # select and add to archive, so that they won't appear in the population again
        amount_to_select = min(ceil(self.population_size * self.selection_proportion), len(self.current_population.current_population))
        selected_individuals = self.current_population.select_and_remove(amount_to_select)
        self.current_population.add_to_archive(selected_individuals)

        # get the neighbourhoods of those selected individuals, add them to the population
        # this is a set to prevent duplicates
        localities = {local for selected_single in selected_individuals
                      for local in self.get_localities(selected_single)}
        localities = self.evaluate(list(localities))  # note the type change

        self.current_population.add_to_population(localities)

    def slow_top(self, how_many: int):
        return heapq.nlargest(how_many, self.current_population.current_population, key=lambda x: x.aggregated_score)

    def show_best_of_current_population(self, how_many: int):
        for individual in self.slow_top(how_many):
            print(individual)

    def run(self,
            termination_criteria: TerminationCriteria,
            show_each_generation=False):
        iteration = 0

        def termination_criteria_met():
            if len(self.current_population.current_population) == 0:
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
        return heapq.nlargest(quantity_returned, self.archive)

    def get_used_evaluations(self) -> int:
        return self.metrics.used_evaluations


def show_plot_of_individuals(individuals: list[Individual], metrics: MultipleMetrics):
    labels = metrics.get_labels()
    points = [i.metric_scores for i in individuals]

    utils.make_interactive_3d_plot(points, labels)


def test_third_archive_miner(problem: BenchmarkProblem,
                             show_each_generation=True):
    print(f"Testing the modified archive miner")
    pRef: PRef = problem.get_pRef(15000)

    budget_limit = AsLongAsWanted()
    # iteration_limit = TerminationCriteria.IterationLimit(12)
    # termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, iteration_limit)

    miner = ThirdArchiveMiner(150,
                              pRef=pRef)

    miner.run(budget_limit, show_each_generation=show_each_generation)

    results = miner.get_results()
    # print("The results of the archive miner are:")
    # for individual in results:
    #     print(f"{individual}")

    print(f"The used budget is {miner.get_used_evaluations()}")

    print("The top 12 by mean fitness are")
    sorted_by_mean_fitness = sorted(results, key=lambda i: i.aggregated_score, reverse=True)
    for individual in sorted_by_mean_fitness[:12]:
        print(individual)
