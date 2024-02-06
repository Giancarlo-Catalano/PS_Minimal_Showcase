import heapq
import warnings
from math import ceil
from typing import TypeAlias

import utils
from BaselineApproaches import Selection
from BaselineApproaches.Evaluator import PSEvaluator
from PRef import PRef
from PS import PS
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria
from BaselineApproaches.Selection import tournament_select

Metrics: TypeAlias = tuple


class ABSM:
    population_size: int
    ps_evaluator: PSEvaluator

    current_population: list[(PS, float)]
    archive: set[PS]

    selection_proportion = 0.3
    tournament_size = 3

    def __init__(self,
                 population_size: int,
                 ps_evaluator: PSEvaluator):
        self.population_size = population_size
        self.ps_evaluator = ps_evaluator

        self.current_population = self.ps_evaluator.evaluate_population(self.make_initial_population())
        self.archive = set()

    @property
    def pRef(self) -> PRef:
        return self.ps_evaluator.pRef

    @property
    def search_space(self) -> SearchSpace:
        return self.ps_evaluator.pRef.search_space

    def make_initial_population(self) -> list[PS]:
        """ basically takes the elite of the PRef, and converts them into PSs """
        """this is called get_init in the paper"""
        return [PS.empty(self.search_space)]

    def get_localities(self, ps: PS) -> list[PS]:
        return ps.specialisations(self.search_space)

    def select_one(self) -> PS:
        return tournament_select(self.current_population, tournament_size=self.tournament_size)

    def top(self, evaluated_population: list[(PS, float)], amount: int) -> list[(PS, float)]:
        return heapq.nlargest(amount, evaluated_population, key=utils.second)

    def make_new_evaluated_population(self):
        selected = [self.select_one() for _ in range(ceil(self.population_size * self.selection_proportion))]
        localities = [local for ps in selected
                      for local in self.get_localities(ps)]
        self.archive.update(selected)

        old_population, _ = utils.unzip(self.current_population)
        new_population = set(old_population).union(localities).difference(self.archive)
        evaluated_new_population = self.ps_evaluator.evaluate_population(list(new_population))

        return self.top(evaluated_new_population, self.population_size)

    def run(self, termination_criteria: TerminationCriteria):
        iteration = 0

        def termination_criteria_met():
            if len(self.current_population) == 0:
                warnings.warn("The run is ending because the population is empty!!!")
                return True

            return termination_criteria.met(iterations=iteration,
                                            evaluations=self.ps_evaluator.used_evaluations,
                                            evaluated_population=self.current_population)

        while not termination_criteria_met():
            self.current_population = self.make_new_evaluated_population()
            iteration += 1


    def get_best_of_last_run(self, quantity_returned: int) -> list[(PS, float)]:
        archive_as_list = list(self.archive)
        evaluated_archive = self.ps_evaluator.evaluate_population(archive_as_list)
        return Selection.top_evaluated(evaluated_archive, quantity_returned)
