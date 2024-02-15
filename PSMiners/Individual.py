from typing import TypeAlias, Optional

import numpy as np

import utils
from PS import PS
from PSMetric.Metric import ManyMetrics
from custom_types import ArrayOfFloats

Metrics: TypeAlias = tuple


class Individual:
    ps: PS
    metrics: Optional[ArrayOfFloats]

    aggregated_score: float

    def __init__(self, ps: PS):
        self.ps = ps
        self.metrics = None
        self.aggregated_score = 0  # dummy value
    def __repr__(self):
        metrics_string = ",".join(f"{m:.3f}" for m in self.metrics)
        return f"{self.ps}, metrics = {metrics_string}, as = {self.aggregated_score:.3f}"

    def calculate_metrics(self, metrics: ManyMetrics):
        self.metrics = np.array(metrics.get_scores(self.ps))




def add_metrics(population: list[Individual], metrics: ManyMetrics) -> list[Individual]:
    for individual in population:
        individual.calculate_metrics(metrics)

    return population


def with_aggregated_scores(population: list[Individual]) -> list[Individual]:
    metrics = np.array([x.metrics for x in population])
    normalised = utils.remap_each_column_in_zero_one(metrics)
    average_for_each_row = np.average(normalised, axis=1)

    for individual, aggregated in zip(population, average_for_each_row):
        individual.aggregated_score = aggregated

    return population
