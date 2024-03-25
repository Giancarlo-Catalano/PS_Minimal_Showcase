import random

import numpy as np
from jmetal.algorithm.multiobjective import NSGAII, MOEAD, MOCell, GDE3
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.lab.visualization import Plot
from jmetal.operator import IntegerPolynomialMutation
from jmetal.operator.crossover import IntegerSBXCrossover, DifferentialEvolutionCrossover
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.neighborhood import C9
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from pandas import DataFrame

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from JMetal.JMetalUtils import into_PS
from JMetal.PSOperator import SpecialisationMutation, BidirectionalMutation, HalfNHalfMutation
from PS import STAR
from PSMetric.Atomicity import Atomicity
from PSMetric.Linkage import Linkage
from PSMetric.LocalPerturbation import BivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import MultipleMetrics
from PSMetric.Simplicity import Simplicity


class PSProblem(IntegerProblem):
    lower_bounds: list[int]
    upper_bounds: list[int]

    many_metrics: MultipleMetrics

    def __init__(self, benchmark_problem: BenchmarkProblem, many_metrics: MultipleMetrics):
        super(PSProblem, self).__init__()

        self.many_metrics = many_metrics
        pRef = benchmark_problem.get_pRef(10000)
        self.many_metrics.set_pRef(pRef)

        self.obj_directions = [self.MAXIMIZE for _ in range(self.many_metrics.get_amount_of_metrics())]
        self.obj_labels = self.many_metrics.get_labels()
        self.lower_bound = [-1 for _ in benchmark_problem.search_space.cardinalities]
        self.upper_bound = [cardinality - 1 for cardinality in benchmark_problem.search_space.cardinalities]

    def number_of_constraints(self) -> int:
        return 0

    def number_of_objectives(self) -> int:
        return self.many_metrics.get_amount_of_metrics()

    def number_of_variables(self) -> int:
        return 1

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        ps = into_PS(solution)
        solution.objectives = [-score for score in self.many_metrics.get_scores(ps)]
        return solution

    def create_random_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints())

        new_solution.variables = [random.randrange(lower, upper + 1)
                                  for lower, upper in zip(self.lower_bound, self.upper_bound)]

        return new_solution

    def create_empty_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints())

        new_solution.variables = [STAR for value in new_solution.lower_bound]
        return new_solution

    def create_solution(self) -> IntegerSolution:
        return self.create_empty_solution()

    def name(self) -> str:
        return "PS search problem"

    @property
    def amount_of_parameters(self) -> int:
        return len(self.lower_bound)


class NormalisedObjectivePSProblem(PSProblem):
    def __init__(self, benchmark_problem: BenchmarkProblem, many_metrics: MultipleMetrics):
        super().__init__(benchmark_problem, many_metrics)

    def number_of_objectives(self) -> int:
        return self.many_metrics.get_amount_of_metrics()

    def number_of_variables(self) -> int:
        return 1

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        ps = into_PS(solution)
        solution.objectives = [-score for score in self.many_metrics.get_normalised_scores(ps)]
        return solution

    def name(self) -> str:
        return "PS search problem (Normalised Objectives)"


class SingleObjectivePSProblem(PSProblem):
    def __init__(self, benchmark_problem: BenchmarkProblem, many_metrics: MultipleMetrics):
        super().__init__(benchmark_problem, many_metrics)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_variables(self) -> int:
        return 1

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        ps = into_PS(solution)
        scores = self.many_metrics.get_normalised_scores(ps)
        solution.objectives[0] = -self.many_metrics.get_aggregated_score(scores)
        return solution

    def name(self) -> str:
        return "PS search problem (Normalised Single Objective)"


def construct_MO_algorithm(which: str,
                           population_size: int,
                           problem: PSProblem,
                           termination_criterion,
                           mutation_operator):
    crossover_operator = IntegerSBXCrossover(probability=0, distribution_index=20)
    if which == "NSGAII":
        return NSGAII(
            problem=problem,
            population_size=population_size,
            offspring_population_size=population_size * 10,
            mutation=mutation_operator,
            crossover=crossover_operator,
            termination_criterion=termination_criterion)
    if which == "NSGAIII":
        reference_directions = UniformReferenceDirectionFactory(n_dim=problem.number_of_objectives(),
                                                                n_points=30)
        return NSGAIII(
            reference_directions=reference_directions,
            problem=problem,
            mutation=mutation_operator,
            crossover=crossover_operator,
            population_size=population_size,
            termination_criterion=termination_criterion)
    elif which == "MOEAD":
        return MOEAD(
            problem=problem,
            population_size=population_size,
            crossover=DifferentialEvolutionCrossover(CR=0.5, F=0.5, K=0.5),
            mutation=mutation_operator,
            aggregative_function=Tschebycheff(dimension=problem.number_of_objectives()),
            neighbor_size=20,
            neighbourhood_selection_probability=0.9,
            max_number_of_replaced_solutions=2,
            weight_files_path='resources/MOEAD_weights',
            termination_criterion=termination_criterion)
    elif which == "MOCell":
        return MOCell(
            problem=problem,
            population_size=population_size,
            neighborhood=C9(10, 10),
            archive=CrowdingDistanceArchive(100),
            mutation=mutation_operator,
            crossover=crossover_operator,
            termination_criterion=termination_criterion)
    elif which == "GDE3":
        return GDE3(problem=problem,
                    population_size=100,
                    cr=0.5,
                    f=0.5,
                    termination_criterion=termination_criterion)
    else:
        raise Exception(f"The algorithm {which} was not recognised")


def test_PSProblem(benchmark_problem: BenchmarkProblem,
                   which_mo_method: str,
                   metrics=None,
                   normalised_objectives=True,
                   show_interactive_plot=False,
                   single_objective=False,
                   save_to_files=False,
                   evaluation_budget=10000):
    if metrics is None:
        metrics = MultipleMetrics([Simplicity(), MeanFitness(), Linkage()])

    if single_objective:
        problem = SingleObjectivePSProblem(benchmark_problem, metrics)
    elif normalised_objectives:
        problem = NormalisedObjectivePSProblem(benchmark_problem, metrics)
    else:
        problem = PSProblem(benchmark_problem, metrics)

    # mutation_operator = SpecialisationMutation(probability=1/problem.amount_of_parameters)
    mutation_operator = HalfNHalfMutation(0.5)
    mutation_operator.ps_mutation_operator.set_search_space(benchmark_problem.search_space)
    termination_criterion = StoppingByEvaluations(evaluation_budget)
    algorithm = construct_MO_algorithm(problem=problem,
                                       which=which_mo_method,
                                       population_size=300,
                                       mutation_operator=mutation_operator,
                                       termination_criterion=termination_criterion)

    print("The metrics are ", metrics.get_labels())

    algorithm.run()

    print("The algorithm has stopped, the results are")

    results = algorithm.get_result()

    metric_labels = metrics.get_labels()
    for item in results:
        ps = into_PS(item)
        objectives = item.objectives
        objectives_string = ", ".join(f"{metric_str} = {-value:.3f}" for metric_str, value in zip(metric_labels, objectives))
        print(f"{ps}, {objectives_string}")

    # save to files

    if save_to_files:
        metric_labels = problem.many_metrics.get_labels()
        filename = f"resources\images\{which_mo_method}_{benchmark_problem}(" + "+".join(metric_labels) + ")"
        print_function_values_to_file(results, filename)
        print_variables_to_file(results, filename)

        # plot
        plot_front = Plot(title='Pareto front approximation', axis_labels=metric_labels)
        plot_front.plot(results, label=filename, filename=filename, format='png')

    if show_interactive_plot:
        labels = metrics.get_labels()
        points = [i.objectives for i in results]
        points = [[-value for value in coords] for coords in points]

        utils.make_interactive_3d_plot(points, labels)


def test_MO(problem: BenchmarkProblem):
    algorithms = ["NSGAII", "MOEAD", "MOCell", "GDE3"]

    for algorithm in algorithms:
        print(f"\n\nTesting with {algorithm}")
        test_PSProblem(problem, which_mo_method=algorithm)


def test_MO_comprehensive(problem: BenchmarkProblem):
    objective_combinations = [
                              # [BivariateLocalPerturbation()],
                              # [Linkage()],
                              [BivariateLocalPerturbation(), MeanFitness()],
                              # [Linkage(), MeanFitness()],
                            ]

    print("Testing with multiple objectives")
    for algorithm in ["NSGAII", "GDE3"]:
        print(f"\n\nTesting with {algorithm}")
        for metrics in objective_combinations:
            test_PSProblem(problem,
                           which_mo_method=algorithm,
                           metrics=MultipleMetrics(metrics),
                           normalised_objectives=False,
                           save_to_files=True,
                           evaluation_budget=15000)

    return
    print("Testing with a single objective")
    for algorithm in ["NSGAII", "GDE3"]:
        print(f"\n\nTesting with {algorithm}")
        test_PSProblem(problem,
                       which_mo_method=algorithm,
                       metrics=MultipleMetrics([MeanFitness(), Linkage()]),
                       single_objective=True,
                       evaluation_budget=15000,
                       save_to_files=False)
