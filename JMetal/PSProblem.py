import random

import numpy as np
from jmetal.algorithm.multiobjective import NSGAII, MOEAD, MOCell, GDE3
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

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from JMetal.JMetalUtils import into_PS
from PSMetric.Atomicity import Atomicity
from PSMetric.KindaAtomicity import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import ManyMetrics
from PSMetric.Simplicity import Simplicity


class PSProblem(IntegerProblem):
    lower_bounds: list[int]
    upper_bounds: list[int]

    many_metrics: ManyMetrics

    def __init__(self, benchmark_problem: BenchmarkProblem, many_metrics: ManyMetrics):
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

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints())

        new_solution.variables = [random.randrange(lower, upper + 1)
                                  for lower, upper in zip(self.lower_bound, self.upper_bound)]

        return new_solution

    def name(self) -> str:
        return "PS search problem"

    @property
    def amount_of_parameters(self) -> int:
        return len(self.lower_bound)


class NormalisedObjectivePSProblem(PSProblem):
    def __init__(self, benchmark_problem: BenchmarkProblem, many_metrics: ManyMetrics):
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
    def __init__(self, benchmark_problem: BenchmarkProblem, many_metrics: ManyMetrics):
        super().__init__(benchmark_problem, many_metrics)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_variables(self) -> int:
        return 1

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        ps = into_PS(solution)
        scores = self.many_metrics.get_normalised_scores(ps)
        solution.objectives[0] = -np.average(scores)
        return solution

    def name(self) -> str:
        return "PS search problem (Normalised Single Objective)"


def make_NSGAII(problem: PSProblem):
    return NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=IntegerPolynomialMutation(probability=1 / problem.amount_of_parameters,
                                           distribution_index=20),
        crossover=IntegerSBXCrossover(probability=0.5, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000))


def make_MOEAD(problem: PSProblem):
    return MOEAD(
        problem=problem,
        population_size=300,
        crossover=DifferentialEvolutionCrossover(CR=0.5, F=0.5, K=0.5),
        mutation=IntegerPolynomialMutation(probability=1.0 / problem.amount_of_parameters,
                                           distribution_index=20),
        aggregative_function=Tschebycheff(dimension=problem.number_of_objectives()),
        neighbor_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        weight_files_path='resources/MOEAD_weights',
        termination_criterion=StoppingByEvaluations(10000)
    )


def make_MOCELL(problem: PSProblem):
    return MOCell(
        problem=problem,
        population_size=100,
        neighborhood=C9(10, 10),
        archive=CrowdingDistanceArchive(100),
        mutation=IntegerPolynomialMutation(probability=1 / problem.amount_of_parameters,
                                           distribution_index=20),
        crossover=IntegerSBXCrossover(probability=0.5, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000)
    )


def make_GDE3(problem: PSProblem):
    return GDE3(problem=problem,
                population_size=100,
                cr=0.5,
                f=0.5,
                termination_criterion=StoppingByEvaluations(10000)
                )


def test_PSProblem(benchmark_problem: BenchmarkProblem,
                   which_mo_method: str,
                   metrics=None,
                   normalised_objectives=True,
                   single_objective=False,
                   save_to_files=False):
    if metrics is None:
        metrics = ManyMetrics([Simplicity(), MeanFitness(), Linkage()])

    if single_objective:
        problem = SingleObjectivePSProblem(benchmark_problem, metrics)
    elif normalised_objectives:
        problem = NormalisedObjectivePSProblem(benchmark_problem, metrics)
    else:
        problem = PSProblem(benchmark_problem, metrics)
    algorithm = None

    if which_mo_method == "NSGAII":
        algorithm = make_NSGAII(problem)
    elif which_mo_method == "MOEAD":
        algorithm = make_MOEAD(problem)
    elif which_mo_method == "MOCell":
        algorithm = make_MOCELL(problem)
    elif which_mo_method == "GDE3":
        algorithm = make_GDE3(problem)
    else:
        raise Exception(f"The algorithm {which_mo_method} was not recognised")

    print("Setup the algorithm, now we run it")

    algorithm.run()

    print("The algorithm has stopped, the results are")

    front = get_non_dominated_solutions(algorithm.get_result())

    for item in front:
        ps = into_PS(item)
        print(f"{ps}, {item.objectives}")

    # save to files

    if save_to_files:
        metric_labels = problem.many_metrics.get_labels()
        filename = f"resources\images\{which_mo_method}_{benchmark_problem}(" + "+".join(metric_labels) + ")"
        print_function_values_to_file(front, filename)
        print_variables_to_file(front, filename)

        # plot
        plot_front = Plot(title='Pareto front approximation', axis_labels=metric_labels)
        plot_front.plot(front, label=filename, filename=filename, format='png')


def test_MO(problem: BenchmarkProblem):
    algorithms = ["NSGAII", "MOEAD", "MOCell", "GDE3"]

    for algorithm in algorithms:
        print(f"\n\nTesting with {algorithm}")
        test_PSProblem(problem, which_mo_method=algorithm)


def test_MO_comprehensive(problem: BenchmarkProblem):
    algorithms = ["NSGAII", "MOEAD", "MOCell", "GDE3"]
    objective_combinations = [[Simplicity(), MeanFitness(), Linkage()],
                              [Simplicity(), MeanFitness()],
                              [MeanFitness(), Linkage()],
                              [Simplicity(), Linkage()]]

    print("Testing with multiple objectives")
    for algorithm in algorithms:
        print(f"\n\nTesting with {algorithm}")
        for metrics in objective_combinations:
            test_PSProblem(problem,
                           which_mo_method=algorithm,
                           metrics=ManyMetrics(metrics),
                           normalised_objectives=True,
                           save_to_files=True)


    print("Testing with a single objective")
    for algorithm in algorithms:
        print(f"\n\nTesting with {algorithm}")
        test_PSProblem(problem,
                       which_mo_method=algorithm,
                       single_objective=True,
                       save_to_files=False)