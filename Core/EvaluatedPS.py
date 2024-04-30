import functools
from typing import Optional

from Core.PS import PS


@functools.total_ordering
class EvaluatedPS:
    ps: PS
    metric_scores: Optional[list[float]]
    # normalised_metric_scores: Optional[list[float]]
    aggregated_score: Optional[float]
    food_score: Optional[float]
    average_of_metrics: Optional[float]

    def __init__(self, ps: PS):
        self.ps = ps
        self.metric_scores = None
        # self.normalised_metric_scores = None
        self.aggregated_score = None
        self.food_score = None
        self.average_of_metrics = None

    def __repr__(self):
        result = f"{self.ps}"
        if self.metric_scores is not None:
            result += "["
            for metric in self.metric_scores:
                result += f"{metric:.3f}, "
            result += "]"
        if self.average_of_metrics is not None:
            result += f", average_of_metrics = {self.average_of_metrics:.3f}"
        if self.food_score is not None:
            result += f", food_score = {self.food_score:.3f}"
        if self.aggregated_score is not None:
            result += f", aggregated_score = {self.aggregated_score:.3f}"
        return result

    def __hash__(self):
        return self.ps.__hash__()

    def __eq__(self, other):
        return self.ps == other.ps

    def __lt__(self, other):
        return self.aggregated_score < other.aggregated_score
