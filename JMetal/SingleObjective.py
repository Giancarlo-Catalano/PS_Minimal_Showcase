import random
from typing import Any

from jmetal.algorithm.singleobjective import EvolutionStrategy, GeneticAlgorithm, LocalSearch, SimulatedAnnealing
from jmetal.core.solution import IntegerSolution
from jmetal.operator import RouletteWheelSelection, IntegerPolynomialMutation
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from JMetal.JMetalUtils import into_PS
from JMetal.PSOperator import BidirectionalMutation, SpecialisationMutation
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
        solution.objectives[0] = 1 - (scores[0] + scores[1]) / 2
        return solution

    def name(self) -> str:
        return "PS search problem (Normalised Single Product Objective)"

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives(),
            number_of_constraints=self.number_of_constraints())

        new_solution.variables = [random.randrange(lower, upper + 1)
                                  for lower, upper in zip(self.lower_bound, self.upper_bound)]

        return new_solution


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
                                population_size=50,
                                offspring_population_size=50,
                                mutation=mutation_operator,
                                crossover=crossover,
                                selection=RouletteWheelSelection(),
                                termination_criterion=termination_criterion)
    elif which == "Local":
        return LocalSearch(problem=problem,
                           mutation=mutation_operator,
                           termination_criterion=termination_criterion)
    elif which == "Annealing":
        return SimulatedAnnealing(problem=problem,
                                  mutation=mutation_operator,
                                  termination_criterion=termination_criterion)
    else:
        raise Exception(f"The requested algorithm \"{which}\" could not be found")


def test_single_objective_search(benchmark_problem: BenchmarkProblem,
                                 evaluation_budget=15000):
    problem = ProductPSProblem(benchmark_problem)

    mutation_probability = 1 / benchmark_problem.search_space.amount_of_parameters
    mutation_operators = [SpecialisationMutation(probability=mutation_probability),
                          BidirectionalMutation(probability=mutation_probability),
                          IntegerPolynomialMutation(probability=mutation_probability)]

    crossover_operators = [IntegerSBXCrossover(probability=0.5, distribution_index=20),
                           IntegerSBXCrossover(probability=0, distribution_index=20),
                           IntegerSBXCrossover(probability=1, distribution_index=20)]



    for algorithm_name in ["Evolutionary", "Genetic", "Local", "Annealing"]:
        for mutation_operator in mutation_operators:
            for crossover_operator in crossover_operators:
                print(f"Run with algorithm = {algorithm_name}, mutation_operator = {mutation_operator.get_name()}, crossover_chance = {crossover_operator.probability}")
                algorithm = get_single_objective_algorithm(problem,
                                                           which=algorithm_name,
                                                           mutation_operator=mutation_operator,
                                                           crossover=crossover_operator,
                                                           termination_criterion=StoppingByEvaluations(evaluation_budget))
                algorithm.run()

                results = algorithm.solutions

                print("The solutions of the run are ")
                for item in results:
                    ps = into_PS(item)
                    print(f"{ps}, {item.objectives}")


                print("The results of the run are")
                print(algorithm.get_result())
