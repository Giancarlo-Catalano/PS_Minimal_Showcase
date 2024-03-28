import random

import numpy as np

import utils
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.LocalPerturbation import BivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric, MultipleMetrics
from PSMetric.Simplicity import Simplicity
from PSMiners.Individual import Individual
from PSMiners.Operators.PSSelectionOperator import TruncationSelection
from PSMiners.PSMiner import PSMiner, Population


# performance issues:

class BaselineArchiveMiner(PSMiner):
    """use this for testing"""
    population_size: int

    archive: set[Individual]

    def __init__(self,
                 population_size: int,
                 metric: MultipleMetrics,
                 pRef: PRef,
                 set_pRef_in_metric = True):
        self.population_size = population_size

        super().__init__(metric=metric,
                         pRef=pRef,
                         set_pRef_in_metric=set_pRef_in_metric)
        self.archive = set()


    def __repr__(self):
        return f"ArchiveMiner(population_size = {self.population_size})"

    def get_initial_population(self) -> Population:
        """ basically takes the elite of the PRef, and converts them into PSs """
        """this is called get_init in the paper"""
        return [Individual(PS.empty(self.search_space))]

    def get_localities(self, individual: Individual) -> list[Individual]:
        return [Individual(ps) for ps in individual.ps.specialisations(self.search_space)]

    def select_one(self) -> Individual:
        tournament_size = 12
        tournament_pool = random.choices(self.current_population, k=tournament_size)
        return max(tournament_pool, key=lambda x: x.aggregated_score)


    def evaluate_individuals(self, newborns: Population) -> Population:
        for individual in newborns:
            individual.metric_scores = self.metric.get_normalised_scores(individual.ps)
        return newborns

    def with_aggregated_scores(self, population: list[Individual]) -> list[Individual]:
        metric_matrix = np.array([ind.metric_scores for ind in population])

        averages = np.average(metric_matrix, axis=1)
        for individual, score in zip(population, averages):
            individual.aggregated_score = score

        return population

    def step(self):
        """ a recreation of the original code"""
        # select and add to archive, so that they won't appear in the population again

        self.current_population = PSMiner.without_duplicates(self.current_population)
        self.current_population = self.evaluate_individuals(self.current_population)
        self.current_population = self.with_aggregated_scores(self.current_population)
        self.current_population = self.get_best_n(n=self.population_size, population=self.current_population)

        parents = self.get_best_n(n = self.population_size // 3, population=self.current_population)
        parents = PSMiner.without_duplicates(parents)
        children = [Individual(child) for parent in parents for child in parent.ps.specialisations(self.search_space)]

        self.archive.update(parents)
        self.current_population.extend(children)
        self.current_population = [ind for ind in self.current_population if ind not in self.archive]

    def get_results(self, quantity_returned: int) -> Population:
        evaluated_archive = self.with_aggregated_scores(list(self.archive))
        return self.get_best_n(n=quantity_returned, population=evaluated_archive)

    def get_parameters_as_dict(self) -> dict:
        return {"kind": "Archive",
                "population_size": self.population_size,
                "metric": repr(self.metric)}


    @classmethod
    def with_default_settings(cls, pRef: PRef):
        return cls(population_size=150,
                   pRef = pRef,
                   metric = MultipleMetrics([Simplicity(), MeanFitness(), BivariateLocalPerturbation()]))



