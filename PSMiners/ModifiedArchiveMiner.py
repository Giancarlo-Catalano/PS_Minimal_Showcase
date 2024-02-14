import heapq
import random
import warnings
from math import ceil
from typing import TypeAlias

import numpy as np

import utils
from BaselineApproaches import Selection
from BaselineApproaches.Evaluator import PSEvaluator
from JMetal.PSProblem import AtomicityEvaluator
from PRef import PRef
from PS import PS
from PSMetric import KindaAtomicity
from PSMetric.KindaAtomicity import LinkageViaMeanFitDiff, SimplerAtomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria
from BaselineApproaches.Selection import tournament_select, tournament_generic
from custom_types import ArrayOfFloats

Metrics: TypeAlias = tuple


class Individual:
    ps: PS
    metrics: ArrayOfFloats

    aggregated_score: float

    def __init__(self, ps: PS, simplicity: float, mean_fitness: float, atomicity: float):
        self.ps = ps
        self.metrics = np.array([simplicity, mean_fitness, atomicity])
        self.aggregated_score = 0  # dummy value

    def __repr__(self):
        return f"{self.ps}, metrics = {self.metrics}, as = {self.aggregated_score}"



class KindaAtomicityEvaluator:
    normalised_pRef: PRef
    global_isolated_benefits: list[list[float]]
    simpler_atomicity_object: SimplerAtomicity

    def __init__(self, pRef: PRef):
        self.simpler_atomicity_object = SimplerAtomicity()
        self.normalised_pRef = self.simpler_atomicity_object.get_normalised_pRef(pRef)
        self.global_isolated_benefits = self.simpler_atomicity_object.get_global_isolated_benefits(self.normalised_pRef)

    def evaluate_single(self, ps: PS) -> float:
        return self.simpler_atomicity_object.get_single_score_using_cached_info(ps,
                                                                                self.normalised_pRef,
                                                                                self.global_isolated_benefits)

class ThreeMetricPSEvaluator:
    simplicity: Simplicity
    mean_fitness: MeanFitness
    linkage_evaluator: AtomicityEvaluator

    pRef: PRef

    evaluations: int

    def __init__(self, pRef: PRef):
        self.pRef = pRef
        self.simplicity = Simplicity()
        self.mean_fitness = MeanFitness()
        self.linkage_evaluator = AtomicityEvaluator(self.pRef)
        self.evaluations = 0

    def evaluate_metrics(self, unevaluated_individuals: list[PS]) -> list[Individual]:
        def get_evaluated(ps: PS):
            simplicity = self.simplicity.get_single_score(ps, self.pRef)
            mean_fitness = self.mean_fitness.get_single_score(ps, self.pRef)
            atomicity = self.linkage_evaluator.evaluate_single(ps)
            return Individual(ps, simplicity, mean_fitness, atomicity)

        self.evaluations += len(unevaluated_individuals)
        return [get_evaluated(ps) for ps in unevaluated_individuals]

    def assign_aggregated_scores(self, population: list[Individual]) -> list[Individual]:
        metrics = np.array([x.metrics for x in population])
        normalised = utils.remap_each_column_in_zero_one(metrics)
        average_for_each_row = np.average(normalised, axis=1)

        for individual, aggregated in zip(population, average_for_each_row):
            individual.aggregated_score = aggregated

        return population


class ModifiedArchiveMiner:
    population_size: int
    pRef: PRef
    evaluator: ThreeMetricPSEvaluator

    current_population: list[Individual]
    archive: set[PS]

    selection_proportion = 0.3
    tournament_size = 3

    def __init__(self,
                 population_size: int,
                 pRef: PRef):
        self.population_size = population_size
        self.pRef = pRef
        self.evaluator = ThreeMetricPSEvaluator(self.pRef)

        self.current_population = self.evaluator.evaluate_metrics(self.make_initial_population())
        self.current_population = self.evaluator.assign_aggregated_scores(self.current_population)
        self.archive = set()

    @property
    def search_space(self) -> SearchSpace:
        return self.pRef.search_space

    def make_initial_population(self) -> list[PS]:
        """ basically takes the elite of the PRef, and converts them into PSs """
        """this is called get_init in the paper"""
        return [PS.empty(self.search_space)]

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

        # evaluate
        new_population = self.evaluator.evaluate_metrics(new_population)
        new_population = self.evaluator.assign_aggregated_scores(new_population)

        return self.top(new_population, self.population_size)

    def make_new_evaluated_population_original(self):
        selected = self.make_selection()
        self.archive.update(selected)
        localities = [local for ps in selected
                      for local in self.get_localities(ps)
                      if local not in self.archive]

        new_population = self.evaluator.evaluate_metrics(localities)
        new_population.extend(individual for individual in self.current_population
                              if individual not in self.archive)

        new_population = list(set(new_population))  # removes duplicates, check if it's necessary
        new_population = self.evaluator.assign_aggregated_scores(new_population)
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
        evaluated_archive = self.evaluator.evaluate_metrics(list(self.archive))
        evaluated_archive = self.evaluator.assign_aggregated_scores(evaluated_archive)
        best = self.top(evaluated_archive, quantity_returned)
        return [(i.ps, i.aggregated_score) for i in best]


    def get_used_evaluations(self) -> int:
        return self.evaluator.evaluations
