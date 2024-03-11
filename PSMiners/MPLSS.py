""" The acronym stands for Mu plus Lambda Scatter search"""

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


class MPLSS:
    current_population: list[Individual]
    mu_parameter: int
    lambda_parameter: int
    diversity_offspring_amount: int
    metric: Metric
    search_space: Optional[SearchSpace]
    mutation_operator: PSMutationOperator

    offspring_amount: int

    def __init__(self,
                 mu_parameter: int,
                 lambda_parameter: int,
                 diversity_offspring_amount: int,
                 metric: Metric,
                 mutation_operator: PSMutationOperator):
        self.mu_parameter = mu_parameter
        self.lambda_parameter = lambda_parameter
        self.metric = metric
        self.diversity_offspring_amount = diversity_offspring_amount
        self.search_space = None
        self.offspring_amount = self.lambda_parameter // self.mu_parameter
        assert (self.lambda_parameter % self.mu_parameter == 0)

        self.current_population = []
        self.mutation_operator = mutation_operator

    def set_pRef(self, pRef: PRef, set_metrics=True):
        if set_metrics:
            self.metric.set_pRef(pRef)
        self.search_space = pRef.search_space
        self.mutation_operator.set_search_space(self.search_space)
        self.current_population = self.get_initial_population(from_uniform=0.33,
                                                              from_half_fixed=0.33,
                                                              from_geometric=0.34)
        self.current_population = self.evaluate_individuals(self.current_population)

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

    def evaluate_individuals(self, individuals: list[Individual]) -> list[Individual]:
        for individual in individuals:
            individual.aggregated_score = self.metric.get_single_normalised_score(individual.ps)
        return individuals

    def truncation_selection(self) -> list[Individual]:
        return heapq.nlargest(n=self.mu_parameter, iterable=self.current_population, key=lambda x: x.aggregated_score)

    def get_offspring(self, individual: Individual) -> list[Individual]:
        return [Individual(self.mutation_operator.mutated(individual.ps))
                for _ in range(self.offspring_amount)]

    def get_fixed_var_proportions(self) -> np.ndarray:
        """The result is an array, where for each variable in the search space we give the proportion
        of the individuals in the population which have that variable fixed"""

        counts = np.zeros(self.search_space.amount_of_parameters, dtype=int)
        for individual in self.current_population:
            counts += individual.ps.values != STAR

        return counts.astype(dtype=float) / len(self.current_population)

    def get_diverse_individuals(self):
        fixed_var_proportions = self.get_fixed_var_proportions()
        sampling_chances = 1 - utils.remap_array_in_zero_one(fixed_var_proportions)

        no_set_values = np.full(shape=self.search_space.amount_of_parameters, fill_value=-1, dtype=int)
        all_var_indexes = list(range(self.search_space.amount_of_parameters))

        def generate_single_individual():
            length = utils.sample_from_geometric_distribution(0.6)
            fixed_positions = random.choices(population=all_var_indexes, weights=sampling_chances, k=length)
            resulting_values = np.copy(no_set_values)
            for position in fixed_positions:
                resulting_values[position] = self.search_space.random_digit(position)
            return Individual(PS(resulting_values))

        return [generate_single_individual() for _ in range(self.diversity_offspring_amount)]

    def run(self,
            termination_criteria: TerminationCriteria,
            custom_ps_repr: Optional[Callable] = None,
            show_each_generation=False):
        iterations = 0

        def should_terminate():
            return termination_criteria.met(iterations=iterations,
                                            used_evaluations=self.metric.used_evaluations)

        while not should_terminate():
            if show_each_generation:
                self.show_current_state(custom_ps_repr)

            # select parents
            selected_parents = self.truncation_selection()

            # make a new population, which contains A: the selected parents (elitism)
            new_population = list(selected_parents)

            for parent in selected_parents:
                # B: contains lambda/mu offsprings for each selected parents
                new_population.extend(self.evaluate_individuals(self.get_offspring(parent)))

            # C: the diversity children
            diverse_children = self.get_diverse_individuals()
            new_population.extend(diverse_children)

            self.current_population = new_population
            # remove duplicates
            self.current_population = list(set(self.current_population))

            iterations += 1

    def get_results(self, quantity_returned=None) -> list[Individual]:
        if quantity_returned == None:
            quantity_returned = self.mu_parameter
        return heapq.nlargest(n=quantity_returned, iterable=self.current_population, key=lambda x: x.aggregated_score)

    def show_current_state(self, custom_ps_repr = None):
        def default_ps_repr(ps):
            return f"{ps}"
        if custom_ps_repr is None:
            custom_ps_repr = default_ps_repr

        amount_to_show = 12
        best_of_population = heapq.nlargest(n=amount_to_show, iterable=self.current_population,
                                            key=lambda x: x.aggregated_score)
        print("\nThe current state is ")
        for best in best_of_population:
            print(f"{custom_ps_repr(best.ps)}, score = {best.aggregated_score:.3f}")


def test_MLPSS_with_MMM(benchmark_problem: BenchmarkProblem):
    print("Testing the MPLSS with the multi modal mutation method")
    print(f"The problem is {benchmark_problem.long_repr()}")

    print("Generating a pRef")
    pRef = benchmark_problem.get_pRef(sample_size=10000)

    metric = Averager([MeanFitness(), Linkage()])
    metric.set_pRef(pRef)
    print("pRef was set")

    def mutation_operator_trial(mutation_operator: PSMutationOperator):
        print(f"Constructing the algorithm with MMMM = {mutation_operator}")
        algorithm = MPLSS(mu_parameter=50,
                          lambda_parameter=300,
                          mutation_operator=mutation_operator,
                          diversity_offspring_amount=100,
                          metric=metric)

        algorithm.set_pRef(pRef, set_metrics=False)

        # print("Running the algorithm")
        termination_criteria = TerminationCriteria.IterationLimit(benchmark_problem.search_space.hot_encoded_length)
        algorithm.run(termination_criteria, show_each_generation=False, custom_ps_repr=benchmark_problem.repr_ps)

        winners = algorithm.get_results(12)
        for winner in winners:
            print(f"{benchmark_problem.repr_ps(winner.ps)}, score = {winner.aggregated_score:.3f}")

        print(f"The used budget is {algorithm.metric.used_evaluations}")

    for rate in range(10, 11):
        mutation_operator = MultimodalMutationOperator(rate / 20)
        mutation_operator_trial(mutation_operator)
