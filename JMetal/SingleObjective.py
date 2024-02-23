from typing import Any

from jmetal.algorithm.singleobjective import EvolutionStrategy, GeneticAlgorithm
from jmetal.core.solution import IntegerSolution
from jmetal.operator import RouletteWheelSelection
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from JMetal.JMetalUtils import into_PS
from JMetal.PSOperator import BidirectionalMutation
from JMetal.PSProblem import PSProblem
from PSMetric.Atomicity import Atomicity
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import MultipleMetrics


class ProductPSProblem(PSProblem):
    def __init__(self, benchmark_problem: BenchmarkProblem):
        metrics = MultipleMetrics([MeanFitness(), Linkage()])
        super().__init__(benchmark_problem, metrics)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_variables(self) -> int:
        return 1

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        ps = into_PS(solution)
        scores = self.many_metrics.get_normalised_scores(ps)
        solution.objectives[0] = -(scores[0] + scores[1])/2
        return solution

    def name(self) -> str:
        return "PS search problem (Normalised Single Product Objective)"


def get_single_objective_algorithm(problem: ProductPSProblem,
                                   which: str,
                                   mutation_operator: Any,
                                   crossover: Any,
                                   termination_criterion: Any):
    if which == "Evolutionary":
        return EvolutionStrategy(problem=problem,
                                 mu=2,
                                 lambda_=2,
                                 elitist=True,
                                 mutation=mutation_operator,
                                 termination_criterion=termination_criterion)
    elif which == "Genetic":
        return GeneticAlgorithm(problem=problem,
                                population_size=150,
                                offspring_population_size=150,
                                mutation=mutation_operator,
                                crossover=crossover,
                                selection=RouletteWheelSelection(),
                                termination_criterion=termination_criterion)


def test_single_objective_search(benchmark_problem: BenchmarkProblem,
                                 evaluation_budget=15000):
    problem = ProductPSProblem(benchmark_problem)
    mutation_operator = BidirectionalMutation(probability=1 / benchmark_problem.search_space.amount_of_parameters)
    crossover = IntegerSBXCrossover(probability=0.5, distribution_index=20)

    algorithm = get_single_objective_algorithm(problem,
                                               which="Genetic",
                                               mutation_operator=mutation_operator,
                                               crossover=crossover,
                                               termination_criterion=StoppingByEvaluations(evaluation_budget))

    algorithm.run()

    results = algorithm.solutions

    print("The results of the run are ")
    for item in results:
        ps = into_PS(item)
        print(f"{ps}, {item.objectives}")
