import heapq
import random
from typing import Callable

import utils
from BaselineApproaches.Evaluator import Individual, FullSolutionEvaluator
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from EDA.FSIndividual import FSIndividual
from FullSolution import FullSolution
from BaselineApproaches.GA import GA
from SearchSpace import SearchSpace
from TerminationCriteria import EvaluationBudgetLimit


class FullSolutionGA(GA):
    search_space: SearchSpace

    def __init__(self, search_space: SearchSpace,
                 mutation_rate: float,
                 crossover_rate: float,
                 elite_size: int,
                 tournament_size: int,
                 population_size: int,
                 fitness_function: Callable,
                 starting_population=None):

        self.search_space = search_space
        super().__init__(mutation_rate=mutation_rate,
                         crossover_rate=crossover_rate,
                         elite_size=elite_size,
                         tournament_size=tournament_size,
                         population_size=population_size,
                         fitness_function=fitness_function,
                         starting_population=starting_population)

    def random_solution(self) -> FullSolution:
        return FullSolution.random(self.search_space)

    def mutated(self, solution: FullSolution) -> FullSolution:
        result_values = solution.values.copy()
        for variable_index, cardinality in enumerate(self.search_space.cardinalities):
            if self.should_mutate():
                result_values[variable_index] = random.randrange(cardinality)
        return FullSolution(result_values)

    def crossed(self, mother: FullSolution, father: FullSolution) -> FullSolution:
        last_index = len(mother)
        start_cut = random.randrange(last_index)
        end_cut = random.randrange(last_index)
        start_cut, end_cut = min(start_cut, end_cut), max(start_cut, end_cut)

        def take_from(donor: FullSolution, start_index, end_index) -> list[int]:
            return list(donor.values[start_index:end_index])

        child_value_list = (take_from(mother, 0, start_cut) +
                            take_from(father, start_cut, end_cut) +
                            take_from(mother, end_cut, last_index))

        return FullSolution(child_value_list)

    def get_results(self, quantity_returned: int):
        return heapq.nlargest(quantity_returned, self.current_population)


def test_FSGA(benchmark_problem: BenchmarkProblem):
    print("Testing the full solution GA")
    algorithm = FullSolutionGA(search_space=benchmark_problem.search_space,
                               mutation_rate=1 / benchmark_problem.search_space.amount_of_parameters,
                               crossover_rate=0.5,
                               elite_size=2,
                               tournament_size=3,
                               population_size=500,
                               fitness_function=benchmark_problem.fitness_function)

    print("Now running the algorithm")
    termination_criterion = EvaluationBudgetLimit(20000)
    algorithm.run(termination_criterion, show_every_generation=True)

    print("The algorithm has terminated, and the results are")
    results = algorithm.get_results(12)

    for individual in results:
        print(f"{individual}")
