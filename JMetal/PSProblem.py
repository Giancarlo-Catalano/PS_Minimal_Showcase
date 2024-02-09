import random

from jmetal.algorithm.multiobjective import NSGAII
from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution
from jmetal.operator import IntegerPolynomialMutation
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity


class AtomicityEvaluator:
    normalised_pRef: PRef
    global_isolated_benefits: list[list[float]]



    def __init__(self, pRef: PRef):
        self.normalised_pRef = Atomicity.get_normalised_pRef(pRef)
        self.global_isolated_benefits = Atomicity.get_global_isolated_benefits(self.normalised_pRef)

    def evaluate_single(self, ps: PS) -> float:
        return Atomicity.get_single_score_knowing_information(ps, self.normalised_pRef, self.global_isolated_benefits)


def into_PS(metal_solution: IntegerSolution) -> PS:
    return PS(metal_solution.variables)


class PSProblem(IntegerProblem):
    lower_bounds: list[int]
    upper_bounds: list[int]

    simplicity_metric: Simplicity
    meanFitness_metric: MeanFitness
    atomicity_evaluator: AtomicityEvaluator

    pRef: PRef

    def __init__(self, benchmark_problem: BenchmarkProblem):
        super(PSProblem, self).__init__()

        self.obj_directions = [self.MAXIMIZE, self.MAXIMIZE, self.MAXIMIZE]
        self.obj_labels = ["Simplicity", "MeanFitness", "Atomicity"]
        self.lower_bound = [-1 for var in benchmark_problem.search_space.cardinalities]
        self.upper_bound = [cardinality - 1 for cardinality in benchmark_problem.search_space.cardinalities]

        self.pRef = benchmark_problem.get_pRef(1000)
        self.simplicity_metric = Simplicity()
        self.meanFitness_metric = MeanFitness()
        self.atomicity_evaluator = AtomicityEvaluator(self.pRef)

    def number_of_constraints(self) -> int:
        return 0

    def number_of_objectives(self) -> int:
        return 3

    def number_of_variables(self) -> int:
        return 1


    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        ps = into_PS(solution)

        solution.objectives[0] = self.simplicity_metric.get_single_unnormalised_score(ps, self.pRef)
        solution.objectives[1] = self.meanFitness_metric.get_single_unnormalised_score(ps, self.pRef)
        solution.objectives[2] = self.atomicity_evaluator.evaluate_single(ps)
        return solution

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints())

        new_solution.variables = [random.randrange(lower, upper)
                                  for lower, upper in zip(self.lower_bound, self.upper_bound)]

        return new_solution

    def name(self) -> str:
        return "PS search problem"



def test_PSProblem(benchmark_problem: BenchmarkProblem):

    problem = PSProblem(benchmark_problem)
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=IntegerPolynomialMutation(probability=1 / benchmark_problem.search_space.amount_of_parameters, distribution_index=20),
        crossover=IntegerSBXCrossover(probability=1, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000))

    print("Setup the algorithm, now we run it")

    algorithm.run()

    print("The algorithm has stopped, the results are")

    front = get_non_dominated_solutions(algorithm.get_result())




    for item in front:
        ps = into_PS(item)
        print(f"{ps} = {item}\n")