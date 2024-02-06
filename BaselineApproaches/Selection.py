import random

import utils
from BaselineApproaches.Evaluator import EvaluatedPopulation, Individual


def tournament_select(evaluated_population: EvaluatedPopulation,
                      tournament_size: int) -> Individual:
    tournament_pool = random.choices(evaluated_population, k=tournament_size)
    winner = max(tournament_pool, key=utils.second)[0]
    return winner