import warnings
from typing import Callable, TypeAlias, Any

import numpy as np

import utils
from PRef import PRef
from PS import PS
from PSMetric.Metric import Metric
from custom_types import Fitness

Individual: TypeAlias = Any
Population: TypeAlias = list[Individual]
EvaluatedIndividual: TypeAlias = (Individual, Fitness)
EvaluatedPopulation = list[EvaluatedIndividual]


class Evaluator:
    fitness_function: Callable
    used_evaluations: int

    def __init__(self):
        self.used_evaluations = 0

    def evaluate_population(self, population: Population) -> EvaluatedPopulation:
        self.used_evaluations += len(population)
        return [(individual, self.fitness_function(individual)) for individual in population]


class FullSolutionEvaluator(Evaluator):
    fitness_function: Callable

    def __init__(self, fitness_function: Callable):
        super().__init__()
        self.fitness_function = fitness_function

    def evaluate_population(self, population: Population) -> EvaluatedPopulation:
        self.used_evaluations += len(population)
        return [(individual, self.fitness_function(individual)) for individual in population]


MetricValues: TypeAlias = np.ndarray


class PSEvaluator(Evaluator):
    metrics: list[Metric]
    pRef: PRef

    cached_metrics: dict[PS, MetricValues]

    def __init__(self, metrics: list[Metric], pRef: PRef):
        super().__init__()
        self.metrics = metrics
        self.pRef = pRef
        self.cached_metrics = dict()

    # a population contains just partial solutions
    # a raw evaluated population contains partial solutions and some scores derived from the metrics
    # a normalised evaluated population contains PSs and a single float

    def evaluate_fresh_population_with_raw_scores(self, pss: list[PS]) -> list[(PS, MetricValues)]:
        # TODO: consider if this is cheating
        self.used_evaluations += len(pss)

        # this is a matrix
        fitnesses_for_each_metric = np.array([metric.get_unnormalised_scores(pss, self.pRef)
                                              for metric in self.metrics])
        metric_tuples = fitnesses_for_each_metric.T

        pss_with_tuples = list(zip(pss, metric_tuples))

        # update cache
        self.cached_metrics.update(pss_with_tuples)
        return pss_with_tuples

    def evaluate_population_with_raw_scores(self, pss: list[PS]) -> list[(PS, MetricValues)]:
        """ Note that the partial solutions returned by this function are NOT in the same order as the input"""
        fresh_pss: list[PS] = []
        stored_pss: list[(PS, MetricValues)] = []
        for ps in pss:
            cached_value = self.cached_metrics.get(ps)
            if cached_value is None:
                fresh_pss.append(ps)
            else:
                stored_pss.append((ps, cached_value))
        evaluated_fresh_pss = self.evaluate_fresh_population_with_raw_scores(fresh_pss)
        return evaluated_fresh_pss + stored_pss

    def evaluate_population(self, pss: list[PS]) -> list[(PS, float)]:
        with_raw_scores = self.evaluate_population_with_raw_scores(pss)
        if len(with_raw_scores) == 0:
            warnings.warn("An evaluated list of PSs appears to be empty")
        pss, metrics = utils.unzip(with_raw_scores)
        metric_array = np.array(metrics)
        normalised_metric_array = utils.remap_each_column_in_zero_one(metric_array)
        averages_for_each_row = np.average(normalised_metric_array, axis=1)
        return list(zip(pss, averages_for_each_row))
