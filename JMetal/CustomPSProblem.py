import random
from typing import Optional

from jmetal.algorithm.multiobjective import NSGAII, MOEAD, MOCell, GDE3, HYPE
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.operator import IntegerPolynomialMutation
from jmetal.operator.crossover import IntegerSBXCrossover, DifferentialEvolutionCrossover
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.neighborhood import C9
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from JMetal.JMetalUtils import into_PS
from JMetal.MetricEvaluators import MetricEvaluator, SimplicityEvaluator, MeanFitnessEvaluator, AtomicityEvaluator
from JMetal.TestProblem import BoringIntegerProblem
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.lab.visualization import Plot


class CustomPSProblem(IntegerProblem):
    lower_bounds: list[int]
    upper_bounds: list[int]

    metric_evaluators: list[MetricEvaluator]
    pRef: PRef

    def __init__(self, benchmark_problem: BenchmarkProblem, metrics: list[MetricEvaluator]):
        super(CustomPSProblem, self).__init__()

        self.metric_evaluators = metrics
        self.obj_directions = [self.MAXIMIZE for _ in metrics]
        self.obj_labels = [f"{metric}" for metric in metrics]
        self.lower_bound = [-1 for _ in benchmark_problem.search_space.cardinalities]
        self.upper_bound = [cardinality - 1 for cardinality in benchmark_problem.search_space.cardinalities]

        self.pRef = benchmark_problem.get_pRef(10000)

        for metric in self.metric_evaluators:
            metric.setup(self.pRef)

    def number_of_constraints(self) -> int:
        return 0

    def number_of_objectives(self) -> int:
        return len(self.metric_evaluators)

    def number_of_variables(self) -> int:
        return 1

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        ps = into_PS(solution)

        for which, metric in enumerate(self.metric_evaluators):
            solution.objectives[which] = -metric.evaluate_single(ps)

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
        return self.pRef.search_space.amount_of_parameters


def make_NSGAII(problem: CustomPSProblem):
    return NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=IntegerPolynomialMutation(probability=1 / problem.amount_of_parameters,
                                           distribution_index=20),
        crossover=IntegerSBXCrossover(probability=0.5, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000))


def make_MOEAD(problem: CustomPSProblem):
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


def make_MOCELL(problem: CustomPSProblem):
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


def make_GDE3(problem: CustomPSProblem):
    return GDE3(problem=problem,
                population_size=100,
                cr=0.5,
                f=0.5,
                termination_criterion=StoppingByEvaluations(10000)
                )


def test_PSProblem_with_2_metrics(benchmark_problem: BenchmarkProblem, which: str):
    algorithm = None
    problem = CustomPSProblem(benchmark_problem, [AtomicityEvaluator(), MeanFitnessEvaluator()])
    if which == "NSGAII":
        algorithm = make_NSGAII(problem)
    elif which == "MOEAD":
        algorithm = make_MOEAD(problem)
    elif which == "MOCell":
        algorithm = make_MOCELL(problem)
    elif which == "GDE3":
        algorithm = make_GDE3(problem)
    else:
        raise Exception(f"The algorithm {which} was not recognised")

    print("Setup the algorithm, now we run it")

    algorithm.run()

    print("The algorithm has stopped, the results are")

    front = get_non_dominated_solutions(algorithm.get_result())

    for item in front:
        ps = into_PS(item)
        print(f"{ps}, {item.objectives}")


    # save to files
    metric_labels = [f"{m}" for m in problem.metric_evaluators]
    filename = f"resources\images\{which}_{benchmark_problem}(" + "+".join(metric_labels) + ")"
    print_function_values_to_file(front, filename)
    print_variables_to_file(front, filename)

    # plot
    plot_front = Plot(title='Pareto front approximation', axis_labels=metric_labels)
    plot_front.plot(front, label=filename, filename=filename, format='png')
