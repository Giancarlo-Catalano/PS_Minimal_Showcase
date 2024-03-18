""" The acronym stands for Mu plus Lambda Limited Resources"""

import heapq
import random
from math import ceil
from typing import Optional, Callable

import numpy as np

import TerminationCriteria
import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PRef import PRef
from PS import PS, STAR
from PSMetric.Averager import Averager
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric
from PSMiners.Individual import Individual
from PSMiners.PSMutationOperator import PSMutationOperator, SinglePointMutation, MultimodalMutationOperator
from SearchSpace import SearchSpace


class MPLLR:
    current_population: Optional[list[Individual]]
    mu_parameter: int
    lambda_parameter: int
    food_weight: float
    metric: Metric
    search_space: Optional[SearchSpace]
    mutation_operator: PSMutationOperator

    offspring_amount: int

    def __init__(self,
                 mu_parameter: int,
                 lambda_parameter: int,
                 food_weight: float,
                 metric: Metric,
                 mutation_operator: PSMutationOperator,
                 starting_population: Optional[list[Individual]] = None):
        self.mu_parameter = mu_parameter
        self.lambda_parameter = lambda_parameter
        self.food_weight = food_weight
        self.metric = metric
        self.search_space = None
        self.offspring_amount = self.lambda_parameter // self.mu_parameter
        assert (self.lambda_parameter % self.mu_parameter == 0)

        if starting_population is not None and len(starting_population) == 0:
            starting_population = None
        self.current_population = starting_population
        self.mutation_operator = mutation_operator

    def set_pRef(self, pRef: PRef, set_metrics=True):
        if set_metrics:
            self.metric.set_pRef(pRef)
        self.search_space = pRef.search_space
        self.mutation_operator.set_search_space(self.search_space)

        if self.current_population is None:  # in case the population was set already
            self.current_population = [Individual(PS.empty(pRef.search_space))]

        # self.get_initial_population(from_uniform=0.33,from_half_fixed=0.33,from_geometric=0.34)
        self.current_population = self.evaluate_individuals_with_metrics(self.current_population)

    def get_initial_population(self,
                               from_uniform: float,
                               from_half_fixed: float,
                               from_geometric: float) -> list[Individual]:

        def uniform_random() -> PS:
            # note the rand(card + 1) - 1, which allows a number from the range [-1, card -1]
            return PS([random.randrange(cardinality + 1) - 1 for cardinality in self.search_space.cardinalities])

        def half_chance_random() -> PS:
            return PS([STAR if random.random() < 0.5
                       else random.randrange(cardinality)
                       for cardinality in self.search_space.cardinalities])

        def geometric_random_with_success_rate(success_chance: float) -> PS:
            total_var_count = self.search_space.amount_of_parameters

            def get_amount_of_fixed_vars():
                # The geometric distribution is simulated using bernoulli trials,
                # where each trial will add a fixed variable onto the ps
                result_count = 0
                while result_count < total_var_count:
                    if random.random() < success_chance:
                        result_count += 1
                    else:
                        break
                return result_count

            vars_to_include = random.choices(list(range(total_var_count)), k=get_amount_of_fixed_vars())

            result_values = [-1 for var in range(total_var_count)]
            for included_var in vars_to_include:
                result_values[included_var] = random.randrange(
                    self.search_space.cardinalities[included_var])
            return PS(result_values)

        def geometric_random():
            return geometric_random_with_success_rate(2 / 3)

        def generate_amount_with_function(generator, proportion):
            amount = ceil(proportion * self.lambda_parameter)
            return [generator() for _ in range(amount)]

        pss = []
        pss.extend(generate_amount_with_function(uniform_random, from_uniform))
        pss.extend(generate_amount_with_function(half_chance_random, from_half_fixed))
        pss.extend(generate_amount_with_function(geometric_random, from_geometric))

        return [Individual(ps) for ps in pss]

    def evaluate_individuals_with_metrics(self, individuals: list[Individual]) -> list[Individual]:
        for individual in individuals:
            individual.metric_scores = [self.metric.get_single_normalised_score(individual.ps), 0]
        return individuals

    def truncation_selection(self) -> list[Individual]:
        return self.top(self.mu_parameter)

    def get_offspring(self, individual: Individual) -> list[Individual]:
        return [Individual(self.mutation_operator.mutated(individual.ps))
                for _ in range(self.offspring_amount)]

    def get_fixed_counts_food_supply(self) -> np.ndarray:
        """The result is an array, where for each variable in the search space we give the proportion
        of the individuals in the population which have that variable fixed"""

        counts = np.zeros(self.search_space.amount_of_parameters, dtype=int)
        for individual in self.current_population:
            counts += individual.ps.values != STAR

        counts = counts.astype(dtype=float)

        return np.divide(1.0, counts, out=np.zeros_like(counts), where = counts!=0)

    def get_food_score(self, individual: Individual, fixed_counts_supply: np.ndarray):

        if individual.ps.is_empty():
            return 0.5

        food_for_each_var = [food for val, food in zip(individual.ps.values, fixed_counts_supply)
                             if val != STAR]
        return np.average(food_for_each_var)


    def introduce_food_scores(self) -> list[Individual]:
        food_rations = self.get_fixed_counts_food_supply()
        for individual in self.current_population:
            individual.metric_scores[1] = self.get_food_score(individual, food_rations)
            metric_score, food_score = individual.metric_scores  #i know, redundant
            individual.aggregated_score = (1-self.food_weight) * metric_score + self.food_weight * food_score
        return self.current_population



    def run(self,
            termination_criteria: TerminationCriteria,
            custom_ps_repr: Optional[Callable] = None,
            show_each_generation=False):
        iterations = 0

        def should_terminate():
            return termination_criteria.met(iterations=iterations,
                                            evaluations=self.metric.used_evaluations)

        while not should_terminate():
            if show_each_generation:
                self.show_current_state(custom_ps_repr)

            # select parents
            selected_parents = self.truncation_selection()

            # make a new population, which contains A: the selected parents (elitism)
            new_population = list(selected_parents)

            for parent in selected_parents:
                # B: contains lambda/mu offsprings for each selected parents
                new_population.extend(self.evaluate_individuals_with_metrics(self.get_offspring(parent)))

            self.current_population = new_population
            # C: remove duplicates
            self.current_population = list(set(self.current_population))

            # D: introduce food scores
            self.introduce_food_scores()



            iterations += 1


    def top(self, quantity_returned: int) -> list[Individual]:
        return heapq.nlargest(n=quantity_returned, iterable=self.current_population)
    def get_results(self, quantity_returned=None) -> list[Individual]:
        if quantity_returned == None:
            quantity_returned = self.mu_parameter
        return self.top(quantity_returned)

    def show_current_state(self, custom_ps_repr = None):
        def default_ps_repr(ps):
            return f"{ps}"
        if custom_ps_repr is None:
            custom_ps_repr = default_ps_repr

        best_of_population = self.top(quantity_returned=12)
        print("\nThe current state is ")
        for best in best_of_population:
            print(f"{custom_ps_repr(best.ps)}, score = {best.aggregated_score:.3f}")


def test_MLPLR(benchmark_problem: BenchmarkProblem):
    print("Testing the MPLSS with the multi modal mutation method")
    print(f"The problem is {benchmark_problem.long_repr()}")

    print("Generating a pRef")
    pRef = benchmark_problem.get_pRef(sample_size=10000)

    metric = Averager([MeanFitness(), Linkage()])
    metric.set_pRef(pRef)
    print("pRef was set")

    mutation_operator = MultimodalMutationOperator(0.5)
    algorithm = MPLLR(mu_parameter=50,
                      lambda_parameter=300,
                      mutation_operator=mutation_operator,
                      food_weight=0.3,
                      metric=metric)

    algorithm.set_pRef(pRef)

    # print("Running the algorithm")
    termination_criteria = TerminationCriteria.IterationLimit(benchmark_problem.search_space.hot_encoded_length)
    algorithm.run(termination_criteria, show_each_generation=False, custom_ps_repr=benchmark_problem.repr_ps)

    winners = algorithm.get_results(12)
    for winner in winners:
        print(f"{benchmark_problem.repr_ps(winner.ps)}, score = {winner.aggregated_score:.3f}")

    print(f"The used budget is {algorithm.metric.used_evaluations}")

