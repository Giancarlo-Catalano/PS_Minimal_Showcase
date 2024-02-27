import random
from typing import Callable

import numpy as np
import pygad

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from JMetal.JMetalUtils import into_PS
from PS import PS, STAR
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import MultipleMetrics


def test_pygad():
    function_inputs = [4, -2, 3.5, 5, -11, -4.7]  # Function inputs.
    desired_output = 44  # Function output.

    def fitness_func(ga_instance, solution, solution_idx):
        output = np.sum(solution * function_inputs)
        return 1.0 / np.abs(output - desired_output)

    num_generations = 50
    num_parents_mating = 4

    fitness_function = fitness_func

    sol_per_pop = 8
    num_genes = len(function_inputs)

    init_range_low = -2
    init_range_high = 5

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    print("Constructing the ga instance")

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes)

    print("Running the GA")
    ga_instance.run()

    print(f"The result of the run is {ga_instance.population}")

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    value = np.sum(solution * function_inputs)
    print(f"Fitness value of the best solution = {solution_fitness}, the resulting value is {value}")
    print(f"Index of the best solution : {solution_idx}")


def continuous_to_ps(values) -> PS:
    return PS(np.array([round(n) for n in values]))


def get_benchmark_problem_PS_fitness_function_and_metrics(benchmark_problem: BenchmarkProblem,
                                                          sample_count: int) -> (Callable, MultipleMetrics):
    pRef = benchmark_problem.get_pRef(sample_count)
    metrics = MultipleMetrics([MeanFitness(), Linkage()])

    metrics.set_pRef(pRef)

    def fitness_func(ga_instance, values: np.ndarray, solution_index) -> float:
        metric_scores = metrics.get_normalised_scores(PS(values))
        return (metric_scores[0] + metric_scores[1]) / 2

    return fitness_func, metrics


def get_gene_space_from_benchmark_problem(benchmark_problem: BenchmarkProblem):
    return [list(range(cardinality)) for cardinality in benchmark_problem.search_space.cardinalities]


def get_initial_population_for_benchmark_problem(benchmark_problem: BenchmarkProblem,
                                                 from_random_uniform: int,
                                                 from_random_half_fixed: int,
                                                 from_random_geometric: int,
                                                 include_empty=True,
                                                 include_targets=True):
    def get_empty():
        return [STAR for var in benchmark_problem.search_space.cardinalities]

    def get_random_uniform():
        return [random.randrange(-1, cardinality) for cardinality in benchmark_problem.search_space.cardinalities]

    def get_random_half_fixed():
        def get_for_cardinality(cardinality: int):
            if random.random() < 0.5:
                return -1
            else:
                return random.randrange(0, cardinality)

        return [get_for_cardinality(cardinality) for cardinality in benchmark_problem.search_space.cardinalities]

    def get_random_geometric():
        total_var_count = benchmark_problem.search_space.amount_of_parameters

        def get_amount_of_fixed_vars():

            # The geometric distribution is simulated using bernoulli trials,
            # where each trial will add a fixed variable onto the ps
            success_probability = 0.5

            result_count = 0
            while result_count < total_var_count:
                if random.random() < success_probability:
                    result_count += 1
                else:
                    break
            return result_count

        vars_to_include = random.choices(list(range(total_var_count)), k=get_amount_of_fixed_vars())

        result_values = get_empty()
        for included_var in vars_to_include:
            result_values[included_var] = random.randrange(benchmark_problem.search_space.cardinalities[included_var])
        return result_values

    def get_targets():
        return [list(ps.values) for ps in benchmark_problem.get_targets()]

    result = []
    if include_empty:
        result.append(get_empty())
    if include_targets:
        result.extend(get_targets())
    if from_random_uniform is not None:
        result.extend([get_random_uniform() for _ in range(from_random_uniform)])
    if from_random_geometric is not None:
        result.extend([get_random_geometric() for _ in range(from_random_geometric)])
    if from_random_half_fixed is not None:
        result.extend([get_random_half_fixed() for _ in range(from_random_half_fixed)])

    return result


def test_pygad_on_benchmark_problem(benchmark_problem: BenchmarkProblem):
    fitness_function, metrics = get_benchmark_problem_PS_fitness_function_and_metrics(benchmark_problem,
                                                                                      10000)

    def stop_when_exceeding_evaluatons(amount_of_evaluations: int):
        """Creates a closure to constructs the termination function"""
        def stopping_criterion(ga_instance):
            if metrics.used_evaluations >= amount_of_evaluations:
                return "stop"
            else:
                print(f"Used {metrics.used_evaluations} evaluations")
                return None
        return stopping_criterion

    num_generations = 50
    num_parents_mating = 4

    sol_per_pop = 8
    num_genes = benchmark_problem.search_space.amount_of_parameters

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    gene_space = get_gene_space_from_benchmark_problem(benchmark_problem)
    initial_population = get_initial_population_for_benchmark_problem(benchmark_problem,
                                                                      include_empty=True,
                                                                      from_random_uniform=0,
                                                                      from_random_geometric=50,
                                                                      from_random_half_fixed=0,
                                                                      include_targets=False)

    print("Constructing the ga instance")

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           initial_population=initial_population,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           gene_type=int,
                           gene_space=gene_space,
                           on_generation=stop_when_exceeding_evaluatons(15000),
                           stop_criteria="saturate_120")

    print("Running the GA")
    ga_instance.run()

    print(f"The result of the run is")
    for item in ga_instance.population:
        print(continuous_to_ps(item))

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"Parameters of the best solution : {solution}")
    ps = PS(solution)
    print(f"Fitness value of the best solution = {solution_fitness}, the resulting ps is {ps}")
    print(f"Index of the best solution : {solution_idx}")
