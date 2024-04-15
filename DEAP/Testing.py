import random

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PS import PS
from PSMetric.LocalPerturbation import BivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from utils import announce
from deap import algorithms, creator, base, tools


def run_deap_for_benchmark_problem(benchmark_problem: BenchmarkProblem):
    print("Starting run_deap_for_benchmark_problem")
    with announce("Generating the pRef"):
        pRef = benchmark_problem.get_reference_population(sample_size=10000)
    metrics = [BivariateLocalPerturbation()]
    for metric in metrics:
        metric.set_pRef(pRef)

    creator.create("FitnessMax", base.Fitness, weights=[1.0])
    creator.create("DEAPPSIndividual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    def random_cell_value() -> int:
        return random.choice([-1, 0, 1])

    def random_ps_values():
        return creator.DEAPPSIndividual(list(PS.random(benchmark_problem.search_space).values))

    def always_empty_ps():
        return creator.DEAPPSIndividual(list(PS.empty(benchmark_problem.search_space).values))

    toolbox.register("attr_cell", random_cell_value)
    toolbox.register("ps_individual",
                     always_empty_ps)

    def evaluate(ps_individual) -> tuple:
        ps = PS(ps_individual)
        return tuple(metric.get_single_score(ps) for metric in metrics)

    ind1 = toolbox.ps_individual()
    print(f"Created the individual {ind1}")
    print(f"Is the fitness valid? {ind1.fitness.valid}")

    ind1.fitness.values = evaluate(ind1)
    print(f"Is the fitness valid? {ind1.fitness.valid}")
    print(ind1.fitness)



    # ## TODO
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    lower_bounds = [-1 for _ in benchmark_problem.search_space.cardinalities]
    upper_bounds = [card-1 for card in benchmark_problem.search_space.cardinalities]
    toolbox.register("mutate", tools.mutUniformInt, low=lower_bounds, up=upper_bounds, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("population", tools.initRepeat, list, toolbox.ps_individual)
    #
    #

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    pop, logbook = algorithms.eaSimple(toolbox.population(n=300),
                                       toolbox,
                                       cxpb=0.5,
                                       mutpb=0.2,
                                       ngen=15,
                                       verbose=True,
                                       stats=stats)


    print("The logbook is")
    print(logbook)

    print("The last population is ")
    for ind in pop:
        ps = PS(ind)
        print(ps)


