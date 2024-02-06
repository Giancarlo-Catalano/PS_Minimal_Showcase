import heapq
import random
from typing import Any

import utils
from BaselineApproaches.Evaluator import EvaluatedPopulation, Individual


def tournament_select(evaluated_population: EvaluatedPopulation,
                      tournament_size: int) -> Individual:
    tournament_pool = random.choices(evaluated_population, k=tournament_size)
    winner = max(tournament_pool, key=utils.second)[0]
    return winner


def top_evaluated(evaluated_population: list[(Any, float)], quantity_returned: int) -> list[(Any, float)]:
    return heapq.nlargest(quantity_returned, evaluated_population, key=utils.second)
