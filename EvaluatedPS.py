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
class EvaluatedPS:
    ps: PS
    metric_scores: Optional[list[float]]
    # normalised_metric_scores: Optional[list[float]]
    aggregated_score: Optional[float]

    def __init__(self, ps: PS):
        self.ps = ps
        self.metric_scores = None
        # self.normalised_metric_scores = None
        self.aggregated_score = None

    def __repr__(self):
        return f"{self.ps}, metrics = {self.metric_scores}, aggregated_score = {self.aggregated_score}"
        # if self.metric_scores:
        #     metrics_string = ",".join(f"{m:.3f}" for m in self.metric_scores)
        #     return f"{self.ps}, metrics = {metrics_string}, as = {self.aggregated_score}"
        # else:
        #     return f"{self.ps}, as = {self.aggregated_score:.3f}"

    def __hash__(self):
        return self.ps.__hash__()

    def __eq__(self, other):
        return self.ps == other.ps

    def __lt__(self, other):
        return self.aggregated_score < other.aggregated_score
