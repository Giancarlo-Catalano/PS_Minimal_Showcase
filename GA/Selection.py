import heapq
import random
from typing import Any, Callable

import utils
from BaselineApproaches.Evaluator import EvaluatedPopulation, Individual


def tournament_select(evaluated_population: EvaluatedPopulation,
                      tournament_size: int) -> Individual:
    tournament_pool = random.choices(evaluated_population, k=tournament_size)
    winner = max(tournament_pool, key=utils.second)[0]
    return winner


def tournament_generic(evaluated_population: list,
                       tournament_size: int,
                       key: Callable):
    tournament_pool = random.choices(evaluated_population, k=tournament_size)
    winner = max(tournament_pool, key=key)[0]
    return winner


def top_evaluated(evaluated_population: list[(Any, float)], quantity_returned: int) -> list[(Any, float)]:
    return heapq.nlargest(quantity_returned, evaluated_population, key=utils.second)
