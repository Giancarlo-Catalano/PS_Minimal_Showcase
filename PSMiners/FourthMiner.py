import heapq
import random
import warnings
from typing import TypeAlias, Callable

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PRef import PRef
from PS import PS
from PSMetric.Averager import Averager
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import MultipleMetrics, Metric
from PSMetric.SignificantlyHighAverage import SignificantlyHighAverage
from PSMiners.Individual import Individual
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria, EvaluationBudgetLimit

# performance issues:
# eq, required by set

Population: TypeAlias = list[Individual]


class FourthMiner:
    population_size: int
    offspring_population_size: int

    current_population: list[Individual]
    archive: set[Individual]
    metric: Metric

    search_space: SearchSpace
    used_evaluations: int

    custom_repr: Callable

    def __init__(self,
                 population_size: int,
                 offspring_population_size: int,
                 metric: Metric,
                 pRef: PRef,
                 custom_ps_repr = None):
        self.metric = metric
        self.used_evaluations = 0
        self.metric.set_pRef(pRef)
        self.search_space = pRef.search_space
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size
        self.current_population = self.get_initial_population()
        self.archive = set()

        self.custom_repr = custom_ps_repr

    def evaluate(self, population: Population) -> Population:
        for individual in population:
            individual.aggregated_score = self.metric.get_single_normalised_score(individual.ps)
        self.used_evaluations += len(population)
        return population

    def get_initial_population(self) -> Population:
        """ basically takes the elite of the PRef, and converts them into PSs """
        """this is called get_init in the paper"""
        return [Individual(PS.empty(self.search_space))]

    def get_localities(self, individual: Individual) -> list[Individual]:
        return [Individual(ps) for ps in individual.ps.specialisations(self.search_space)]

    def select_one(self) -> Individual:
        tournament_size = 3
        tournament_pool = random.choices(self.current_population, k=tournament_size)
        return max(tournament_pool, key=lambda x: x.aggregated_score)

    def update_population(self):
        """Note how this relies on the current population being evaluated,
        and the new population will also be evaluated"""
        # select and add to archive, so that they won't appear in the population again
        offspring = set()

        remaining_population = set(self.current_population)
        while len(offspring) < self.offspring_population_size and len(remaining_population) > 0:
            selected = self.select_one()
            remaining_population.discard(selected)
            if selected not in self.archive:
                self.archive.add(selected)
                offspring.update(self.get_localities(selected))

        self.current_population = list(remaining_population)
        self.current_population.extend(self.evaluate(list(offspring)))
        self.current_population = self.top(self.population_size)

    def top(self, how_many: int):
        return heapq.nlargest(how_many, self.current_population, key=lambda x: x.aggregated_score)

    def show_best_of_current_population(self, how_many: int):
        def show_ps_default(ps):
            return f"{ps}"

        show_ps = show_ps_default if self.custom_repr is None else self.custom_repr

        for individual in self.top(how_many):
            metrics = self.metric.get_normalised_scores(individual.ps)
            metrics_str = ", ".join(f"{value:.3f}" for value in metrics)
            print(f"{show_ps(individual.ps)}, aggr = {individual.aggregated_score:.3f}, scores = {metrics_str}")

    def run(self,
            termination_criteria: TerminationCriteria,
            show_each_generation=False):
        iteration = 0

        def termination_criteria_met():
            if len(self.current_population) == 0:
                warnings.warn("The run is ending because the population is empty!!!")
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
        return heapq.nlargest(quantity_returned, self.archive, key=lambda x: x.aggregated_score)

    def get_used_evaluations(self) -> int:
        return self.used_evaluations


def show_plot_of_individuals(individuals: list[Individual], metrics: MultipleMetrics):
    labels = metrics.get_labels()
    points = [i.metric_scores for i in individuals]

    utils.make_interactive_3d_plot(points, labels)


def test_fourth_archive_miner(problem: BenchmarkProblem,
                              show_each_generation=True):
    print(f"Testing the modified archive miner")
    print(f"The problem is {problem.long_repr()}")
    pRef: PRef = problem.get_pRef(10000)

    budget_limit = EvaluationBudgetLimit(17000)
    # iteration_limit = TerminationCriteria.IterationLimit(12)
    # termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, iteration_limit)

    miner = FourthMiner(150,
                        offspring_population_size=300,
                        pRef=pRef,
                        metric=Averager([MeanFitness(), Linkage()]),
                        custom_ps_repr=problem.repr_ps)

    miner.run(budget_limit, show_each_generation=show_each_generation)

    results = miner.get_results()
    print(f"The used budget is {miner.get_used_evaluations()}")

    print("The top 12 by mean fitness are")
    sorted_by_mean_fitness = sorted(results, key=lambda i: i.aggregated_score, reverse=True)
    for individual in sorted_by_mean_fitness[:12]:
        print(individual)
