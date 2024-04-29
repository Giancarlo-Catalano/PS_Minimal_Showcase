import TerminationCriteria
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from EvaluatedFS import EvaluatedFS
from GA.GA import GA
from GA.Operators import SinglePointFSMutation, TwoPointFSCrossover, FSSelectionOperator, TournamentSelection
from GA.SA import SA
from PRef import PRef
from utils import announce


def uniformly_random_distribution_pRef(benchmark_problem: BenchmarkProblem,
                                       sample_size: int) -> PRef:
    return benchmark_problem.get_reference_population(sample_size=sample_size)


def pRef_from_GA(benchmark_problem: BenchmarkProblem,
                 ga_population_size: int,
                 sample_size: int) -> PRef:
    algorithm = GA(search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   crossover_operator=TwoPointFSCrossover(),
                   selection_operator=TournamentSelection(),
                   crossover_rate=0.5,
                   elite_proportion=0.02,
                   tournament_size=3,
                   population_size=ga_population_size,
                   fitness_function=benchmark_problem.fitness_function)

    solutions : list[EvaluatedFS] = []
    solutions.extend(algorithm.current_population)

    while len(solutions) < sample_size:
        algorithm.step()
        solutions.extend(algorithm.current_population)

    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_SA(benchmark_problem: BenchmarkProblem,
                 sample_size: int,
                 max_trace: int) -> PRef:
    algorithm = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   cooling_coefficient=0.9995)

    solutions : list[EvaluatedFS] = []

    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace= max_trace))

    solutions = solutions[:sample_size]

    best_solution = max(solutions)
    # df = benchmark_problem.details_of_solution(best_solution.full_solution)   # Experimental
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)
