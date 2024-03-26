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
from PSMiners.PSMiner import PSMiner
from SearchSpace import SearchSpace
import TerminationCriteria


class MuPlusLambda(PSMiner):
    current_population: list[Individual]
    mu_parameter: int
    lambda_parameter: int

    offspring_amount: int  # self.lambda_parameter // self.mu_parameter

    def __init__(self,
                 mu_parameter: int,
                 lambda_parameter: int,
                 metric: Metric,
                 mutation_operator: PSMutationOperator,
                 selection_operator: PSSelectionOperator,
                 pRef: PRef):
        super().__init__(metric=metric,
                         pRef=pRef,
                         mutation_operator=mutation_operator,
                         crossover_operator=None,
                         selection_operator=selection_operator)

        self.mu_parameter = mu_parameter
        self.lambda_parameter = lambda_parameter
        self.offspring_amount = self.lambda_parameter // self.mu_parameter
        assert(self.lambda_parameter % self.mu_parameter == 0)

        self.current_population = []
        self.mutation_operator = mutation_operator
        self.selection_operator = selection_operator

        self.metric.set_pRef(pRef)
        self.current_population = self.get_initial_population()
        self.current_population = self.evaluate_individuals(self.current_population)




    def get_initial_population(self):
        return PSMiner.get_mixed_initial_population(search_space= self.search_space,
                                                    from_uniform=0.33,
                                                    from_geometric=0.33,
                                                    from_half_fixed=0.34,
                                                    population_size=self.lambda_parameter)


    def get_offspring(self, individual: Individual) -> list[Individual]:
        return [Individual(self.mutation_operator.mutated(individual.ps))
                for _ in range(self.offspring_amount)]


    def step(self):
        selected_parents = self.selection_operator.select_n(self.mu_parameter, self.current_population)

        children = []
        for parent in selected_parents:
            children.extend(self.get_offspring(parent))

        children = self.evaluate_individuals(children)

        self.current_population = PSMiner.without_duplicates(selected_parents + children)



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
