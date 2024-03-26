import heapq
import random
from math import ceil
from typing import Optional

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PRef import PRef
from PS import PS, STAR
from PSMetric.Averager import Averager
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric
from PSMiners.Individual import Individual
from PSMiners.Operators.PSMutationOperator import PSMutationOperator, SinglePointMutation, MultimodalMutationOperator
from PSMiners.Operators.PSSelectionOperator import PSSelectionOperator
from SearchSpace import SearchSpace
import TerminationCriteria


class MuPlusLambda:
    current_population: list[Individual]
    mu_parameter: int
    lambda_parameter: int
    metric: Metric
    search_space: Optional[SearchSpace]
    mutation_operator: PSMutationOperator
    selection_operator: PSSelectionOperator

    offspring_amount: int

    def __init__(self,
                 mu_parameter: int,
                 lambda_parameter: int,
                 metric: Metric,
                 mutation_operator: PSMutationOperator,
                 selection_operator: PSSelectionOperator,
                 search_space: SearchSpace,
                 pRef: PRef):
        self.mu_parameter = mu_parameter
        self.lambda_parameter = lambda_parameter
        self.metric = metric
        self.search_space = search_space
        self.offspring_amount = self.lambda_parameter // self.mu_parameter
        assert(self.lambda_parameter % self.mu_parameter == 0)

        self.current_population = []
        self.mutation_operator = mutation_operator
        self.selection_operator = selection_operator

        self.metric.set_pRef(pRef)
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

    def get_offspring(self, individual: Individual) -> list[Individual]:
        return [Individual(self.mutation_operator.mutated(individual.ps))
                for _ in range(self.offspring_amount)]

    def run(self, termination_criteria: TerminationCriteria):
        iterations = 0

        def should_terminate():
            return termination_criteria.met(iterations=iterations,
                                            used_evaluations=self.metric.used_evaluations)

        while not should_terminate():
            selected_parents = self.selection_operator.select_n(self.mu_parameter, self.current_population)

            new_population = list(selected_parents)
            for parent in selected_parents:
                new_population.extend(self.evaluate_individuals(self.get_offspring(parent)))

            self.current_population = new_population
            self.current_population = list(set(self.current_population))

            iterations += 1

    def get_results(self, amount_to_return=None) -> list[Individual]:
        if amount_to_return == None:
            amount_to_return = self.mu_parameter
        return heapq.nlargest(n=amount_to_return, iterable=self.current_population, key=lambda x: x.aggregated_score)


def test_mu_plus_lambda(benchmark_problem: BenchmarkProblem):
    print("Testing the mu plus lambda algorithm")
    print(f"The problem is {benchmark_problem.long_repr()}")

    print("Generating a pRef")
    pRef = benchmark_problem.get_pRef(sample_size=10000)
    mutation_operator = SinglePointMutation(probability=1 / pRef.search_space.amount_of_parameters,
                                            chance_of_unfixing=0.5)

    print("Constructing the algorithm")
    algorithm = MuPlusLambda(mu_parameter=30,
                             lambda_parameter=150,
                             mutation_operator=mutation_operator,
                             metric=Averager([MeanFitness(), Linkage()]))


    print("Running the algorithm")
    termination_criteria = TerminationCriteria.IterationLimit(12)
    algorithm.run(termination_criteria)

    print("Run has terminated, the results are")
    for individual in algorithm.get_results():
        print(f"{individual.ps}, score = {individual.aggregated_score:.3f}")


def test_mu_plus_lambda_with_repeated_trials(benchmark_problem: BenchmarkProblem,
                                             trials: int):
    print("Testing the mu plus lambda algorithm with repeated trials")
    print(f"The problem is {benchmark_problem.long_repr()}")

    print("Generating a pRef")
    pRef = benchmark_problem.get_pRef(sample_size=10000)
    mutation_operator = SinglePointMutation(probability=1 / pRef.search_space.amount_of_parameters,
                                            chance_of_unfixing=0.5)

    metric = Averager([MeanFitness(), Linkage()])
    metric.set_pRef(pRef)

    def single_trial() -> Individual:
        # print("Constructing the algorithm")
        algorithm = MuPlusLambda(mu_parameter=12,
                                 lambda_parameter=60,
                                 mutation_operator=mutation_operator,
                                 metric=metric)


        # print("Running the algorithm")
        termination_criteria = TerminationCriteria.IterationLimit(benchmark_problem.search_space.hot_encoded_length)
        algorithm.run(termination_criteria)

        return max(algorithm.get_results(), key=lambda x: x.aggregated_score)

    winners = []
    for trial in range(trials):
        print(f"Starting trial #{trial}")
        winners.append(single_trial())

    for winner in winners:
        print(f"{benchmark_problem.repr_ps(winner.ps)}, score = {winner.aggregated_score:.3f}\n")


def test_mu_plus_lambda_with_MMM(benchmark_problem: BenchmarkProblem):
    print("Testing the mu plus lambda algorithm with the multi modal mutation method")
    print(f"The problem is {benchmark_problem.long_repr()}")

    print("Generating a pRef")
    pRef = benchmark_problem.get_pRef(sample_size=10000)

    metric = Averager([MeanFitness(), Linkage()])
    metric.set_pRef(pRef)
    print("pRef was set")

    def mutation_operator_trial(mutation_operator: PSMutationOperator):
        print(f"Constructing the algorithm with MMMM = {mutation_operator}")
        algorithm = MuPlusLambda(mu_parameter=50,
                                 lambda_parameter=300,
                                 mutation_operator=mutation_operator,
                                 metric=metric)

        algorithm.set_pRef(pRef, set_metrics=False)

        # print("Running the algorithm")
        termination_criteria = TerminationCriteria.IterationLimit(benchmark_problem.search_space.hot_encoded_length)
        algorithm.run(termination_criteria)

        winners = algorithm.get_results(12)
        for winner in winners:
            print(f"{benchmark_problem.repr_ps(winner.ps)}, score = {winner.aggregated_score:.3f}")

        print(f"The used budget is {algorithm.metric.used_evaluations}")

    for rate in range(20):
        mutation_operator = MultimodalMutationOperator(rate / 20)
        mutation_operator_trial(mutation_operator)
