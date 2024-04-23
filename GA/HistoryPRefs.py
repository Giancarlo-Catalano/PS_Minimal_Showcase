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
    with announce(f"Generating a randomly uniform pRef of size {sample_size}"):
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

    with announce(f"Running a GA with population {ga_population_size} to generate a pRef of size {sample_size}"):
        while len(solutions) < sample_size:
            algorithm.step()
            solutions.extend(algorithm.current_population)

    print(f"The GA used {algorithm.evaluator.used_evaluations} evaluations")

    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_SA(benchmark_problem: BenchmarkProblem,
                 sample_size: int,
                 max_trace: int) -> PRef:
    algorithm = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space))

    solutions : list[EvaluatedFS] = []

    with announce(f"Running a SA to generate a pRef of size {sample_size}"):
        while len(solutions) < sample_size:
            solutions.extend(algorithm.get_one_with_attempts(max_trace= max_trace))

    solutions = solutions[:sample_size]

    print(f"The SA used {algorithm.evaluator.used_evaluations} evaluations")

    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)
