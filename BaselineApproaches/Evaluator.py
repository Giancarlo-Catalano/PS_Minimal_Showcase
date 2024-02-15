import logging
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



