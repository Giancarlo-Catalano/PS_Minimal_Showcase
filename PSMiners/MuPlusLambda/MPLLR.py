""" The acronym stands for Mu plus Lambda Limited Resources"""

import heapq
import random
from math import ceil
from typing import Optional, Callable

import numpy as np

import TerminationCriteria
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from EvaluatedPS import EvaluatedPS
from PRef import PRef
from PS import PS, STAR
from PSMetric.Averager import Averager
from PSMetric.Linkage import Linkage
from PSMetric.LocalPerturbation import BivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric
from PSMiners.MuPlusLambda.MuPlusLambda import MuPlusLambda
from PSMiners.Operators.PSMutationOperator import PSMutationOperator, MultimodalMutationOperator
from PSMiners.Operators.PSSelectionOperator import PSSelectionOperator, TruncationSelection
from PSMiners.AbstractPSMiner import Population, AbstractPSMiner
from SearchSpace import SearchSpace


class MPLLR(MuPlusLambda):
    food_weight: float

    def __init__(self,
                 mu_parameter: int,
                 lambda_parameter: int,
                 pRef: PRef,
                 food_weight: float,
                 metric: Metric,
                 mutation_operator: PSMutationOperator,
                 selection_operator: PSSelectionOperator,
                 seed_population=None,
                 set_pRef_in_metric = True):
        self.food_weight = food_weight
        super().__init__(metric=metric,
                         pRef=pRef,
                         mutation_operator=mutation_operator,
                         selection_operator=selection_operator,
                         seed_population=seed_population,
                         lambda_parameter=lambda_parameter,
                         mu_parameter=mu_parameter,
                         set_pRef_in_metric=set_pRef_in_metric)

    def __repr__(self):
        return (f"MPLLR(mu={self.mu_parameter}, "
                f"lambda={self.lambda_parameter}, "
                f"mutation = {self.mutation_operator}, "
                f"selection = {self.selection_operator}, "
                f"food_weight = {self.food_weight})")

    def evaluate_individuals(self, newborns: Population) -> Population:
        """In order for the food scores to be calculated, the average of the metrics needs to be stored NOT in aggregated_score"""
        for individual in newborns:
            individual.metric_scores = [metric.get_single_normalised_score(individual.ps) for metric in self.metric.metrics]
            individual.average_of_metrics = np.average(individual.metric_scores)
        return newborns

    def get_fixed_counts_food_supply(self) -> np.ndarray:
        """The result is an array, where for each variable in the search space we give the proportion
        of the individuals in the population which have that variable fixed"""

        counts = np.zeros(self.search_space.amount_of_parameters, dtype=int)
        for individual in self.current_population:
            counts += individual.ps.values != STAR

        counts = counts.astype(dtype=float)

        return np.divide(1.0, counts, out=np.zeros_like(counts), where=counts != 0)

    def get_food_score(self, individual: EvaluatedPS, fixed_counts_supply: np.ndarray):

        if individual.ps.is_empty():
            return 0.5

        food_for_each_var = [food for val, food in zip(individual.ps.values, fixed_counts_supply)
                             if val != STAR]
        return np.average(food_for_each_var)

    def introduce_food_scores(self) -> list[EvaluatedPS]:
        food_rations = self.get_fixed_counts_food_supply()
        for individual in self.current_population:
            individual.food_score = self.get_food_score(individual, food_rations)
            individual.aggregated_score = (
                                                      1 - self.food_weight) * individual.average_of_metrics + self.food_weight * individual.food_score
        return self.current_population

    def step(self):
        self.introduce_food_scores()
        super().step()  # looks dodgy but it should work


    def get_parameters_as_dict(self) -> dict:
        result = super().get_parameters_as_dict()
        result["kind"] = "MPLLR"
        result["food_weight"] = self.food_weight
        return result

    @classmethod
    def with_default_settings(cls, pRef: PRef):
        return cls(mu_parameter=50,
                   lambda_parameter=300,
                   metric=Averager([MeanFitness(), BivariateLocalPerturbation()]),
                   mutation_operator=MultimodalMutationOperator(0.5, pRef.search_space),
                   selection_operator=TruncationSelection(),
                   pRef=pRef,
                   food_weight=0.3)

