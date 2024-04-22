from itertools import chain
from operator import attrgetter

import numpy as np
from deap.tools import sortNondominated, sortLogNondominated

from PS import STAR
from SearchSpace import SearchSpace


def gc_selNSGA2(individuals, k, nd='standard'):
    """Apply NSGA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *k* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *individuals*. For more
    details on the NSGA-II operator see [Deb2002]_.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :returns: A list of selected individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """
    if nd == 'standard':
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    for front in pareto_fronts:
        gc_assignCrowdingDist(front)

    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(pareto_fronts[-1], key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen


def get_fixed_counts_food_supply(population, search_space: SearchSpace) -> np.ndarray:
    """The result is an array, where for each variable in the search space we give the proportion
    of the individuals in the population which have that variable fixed"""

    counts = np.zeros(search_space.amount_of_parameters, dtype=int)
    for individual in population:
        counts += np.array(individual) != STAR

    counts = counts.astype(dtype=float)

    return np.divide(1.0, counts, out=np.zeros_like(counts), where=counts != 0)

def get_food_score(self, individual: Individual, fixed_counts_supply: np.ndarray):

    if individual.ps.is_empty():
        return 0.5

    food_for_each_var = [food for val, food in zip(individual.ps.values, fixed_counts_supply)
                         if val != STAR]
    return np.average(food_for_each_var)

def gc_assignCrowdingDist(individuals):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(individuals) == 0:
        return

