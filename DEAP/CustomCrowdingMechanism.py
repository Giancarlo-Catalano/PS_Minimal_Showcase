from itertools import chain
from operator import attrgetter

import numpy as np
from deap.tools import sortNondominated, sortLogNondominated

from PS import STAR
from SearchSpace import SearchSpace

def get_food_supplies(population) -> np.ndarray:
    """The result is an array, where for each variable in the search space we give the proportion
    of the individuals in the population which have that variable fixed"""
    counts = np.sum([individual.values != STAR for individual in population], dtype=float, axis=0)
    return np.divide(1.0, counts, out=np.zeros_like(counts), where=counts != 0)

def get_food_score(individual, fixed_counts_supply: np.ndarray):
    if individual.is_empty():
        return 0 #np.average(fixed_counts_supply)  # very arbitrary to be honest
    return np.average([food for val, food in zip(individual, fixed_counts_supply)
                         if val != STAR])

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
    # this is new, read the comments below


    if nd == 'standard':
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == 'log':
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception('selNSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    # usually, here you would assign a crowing distance like so:
    # for front in pareto_fronts:
    #  gc_assignCrowdingDist(front)

    # instead, we ignore the fronts and assign our own crowding at the start


    chosen = list(chain(*pareto_fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        last_pareto_front = pareto_fronts[-1]
        food_supply = get_food_supplies(individuals)  #note: on the entire population, not on the front
        for individual in last_pareto_front:
            individual.fitness.crowding_dist = get_food_score(individual, food_supply)
        sorted_front = sorted(last_pareto_front, key=attrgetter("fitness.crowding_dist"), reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen




def gc_assignCrowdingDist(population):
    """Assign a crowding distance to each individual's fitness. The
    crowding distance can be retrieve via the :attr:`crowding_dist`
    attribute of each individual's fitness.
    """
    if len(population) == 0:
        return

    food_supply = get_food_supplies(population)
    for individual in population:
        individual.fitness.crowding_dist = get_food_score(individual, food_supply)

