import functools
import random
from typing import TypeAlias, Optional

import numpy as np

import utils
from PS import PS
from PSMetric.Metric import MultipleMetrics
from custom_types import ArrayOfFloats

Metrics: TypeAlias = tuple


@functools.total_ordering
class Individual:
    ps: PS
    metric_scores: Optional[ArrayOfFloats]
    average_of_metrics: float
    food_score: float  # just makes my life a lot easier in LR

    aggregated_score: float

    def __init__(self, ps: PS):
        self.ps = ps
        self.metric_scores = None
        self.aggregated_score = 0  # dummy value

    def __repr__(self):

        if self.metric_scores:
            metrics_string = ",".join(f"{m:.3f}" for m in self.metric_scores)
            return f"{self.ps}, metrics = {metrics_string}, as = {self.aggregated_score:.3f}"
        else:
            return f"{self.ps}, as = {self.aggregated_score:.3f}"

    def calculate_metrics(self, metrics: MultipleMetrics):
        self.metric_scores = np.array(metrics.get_scores(self.ps))

    def calculate_normalised_metrics(self, metrics: MultipleMetrics):
        self.metric_scores = np.array(metrics.get_normalised_scores(self.ps))

    def __hash__(self):
        return self.ps.__hash__()

    def __eq__(self, other):
        return self.ps == other.ps

    def __lt__(self, other):
        return self.aggregated_score < other.aggregated_score


def add_metrics(population: list[Individual], metrics: MultipleMetrics) -> list[Individual]:
    for individual in population:
        individual.calculate_metrics(metrics)

    return population


def add_normalised_metrics(population: list[Individual], metrics: MultipleMetrics) -> list[Individual]:
    for individual in population:
        individual.calculate_normalised_metrics(metrics)

    return population


def with_aggregated_scores(population: list[Individual]) -> list[Individual]:
    metrics = np.array([x.metric_scores for x in population])
    normalised = utils.remap_each_column_in_zero_one(metrics)
    average_for_each_row = np.average(normalised, axis=1)

    for individual, aggregated in zip(population, average_for_each_row):
        individual.aggregated_score = aggregated

    return population


def with_average_score(population: list[Individual]) -> list[Individual]:
    for individual in population:
        individual.aggregated_score = np.average(individual.metric_scores)

    return population


def with_product_score(population: list[Individual]) -> list[Individual]:
    for individual in population:
        individual.aggregated_score = np.product(individual.metric_scores)

    return population


def partition_by_simplicity(population: list[Individual]) -> list[list[Individual]]:
    if len(population) == 0:
        return []

    amount_of_variables = len(population[0].ps)
    result = [[] for _ in range(amount_of_variables + 1)]

    for individual in population:
        amount_of_fixed_vars = individual.ps.fixed_count()
        result[amount_of_fixed_vars].append(individual)

    return result


def tournament_select(individuals: list[Individual], tournament_size: int) -> Individual:
    tournament_pool = random.choices(individuals, k=tournament_size)
    return max(tournament_pool, key=lambda x: x.aggregated_score)
