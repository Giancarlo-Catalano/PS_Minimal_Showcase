
"""This file is born out of the need to make a data stracture containing the population of the archive miner,
with the following requirements:
    * adding individuals to the population, but not letting in the ones in the archive
    * randomly sampling individuals
    * only keeping the best n individuals
    * adding individuals to the archive
"""


import bisect
import heapq
import random

import utils
from PSMiners.Individual import Individual


class EfficientPopulation:
    archive: set[Individual]
    current_population: list[Individual]
    current_worst_individual_score: float
    tournament_size: int

    max_size: int


    def __init__(self,
                 initial_population: list[Individual],
                 max_size: int,
                 tournament_size: int):
        self.archive = set()
        self.current_population = initial_population
        self.tournament_size = tournament_size

        self.max_size = max_size


    def add_to_archive(self, individuals: set[Individual]):
        """ Updates the archive with the given individuals"""
        self.archive = individuals


    def select_and_remove(self, quantity: int):
        """Selects some individuals from the current population using tournament selection,
        and removes them from the population"""

        current_pop_size = len(self.current_population)
        indexes_with_scores = [(index, ind.aggregated_score) for (index, ind) in enumerate(self.current_population)]
        indexes_with_scores.sort(key=utils.second)
        just_indexes = [pair[0] for pair in indexes_with_scores]

        def single_tournament_selection():
            """Returns an index"""
            random_positions = [random.randrange(current_pop_size) for _ in range(self.tournament_size)]
            winner = max(random_positions)
            return just_indexes[winner]

        winning_indexes = [single_tournament_selection() for _ in range(quantity)]
        winning_indexes.sort()

        selected = [self.current_population[index] for index in winning_indexes]
        new_population = []
        starting_point = 0
        for selected_index in winning_indexes:
            new_population.extend(self.current_population[starting_point:selected_index])
            starting_point = selected_index
        self.current_population = new_population
        return selected

    def add_to_population(self, new_individuals: list[Individual]):
        """Adds the given individuals to the population,
            excluding those that are in the archive,
            and limiting the max size to be self.max_size"""

        """assumes that the current population contains nothing from the archive"""

        all_scores = [ind.aggregated_score for ind in self.current_population]
        all_scores.extend([ind.aggregated_score for ind in new_individuals if ind not in self.archive])

        worst_acceptable = heapq.nlargest(self.max_size, all_scores)[-1]

        self.current_population = [ind for ind in self.current_population if ind.aggregated_score >= worst_acceptable]
        self.current_population.extend(ind for ind in new_individuals if ind.aggregated_score >= worst_acceptable)



