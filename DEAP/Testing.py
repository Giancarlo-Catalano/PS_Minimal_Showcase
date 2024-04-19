import random

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from EvaluatedPS import EvaluatedPS
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.GlobalPerturbation import BivariateGlobalPerturbation
from PSMetric.Linkage import Linkage
from PSMetric.LocalPerturbation import BivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric
from PSMetric.Simplicity import Simplicity
from utils import announce
from deap import algorithms, creator, base, tools


def nsgaiii(toolbox,
            stats,
            mu,
            ngen,
            cxpb,
            mutpb):
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=mu)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, mu)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook

def run_deap_for_benchmark_problem(benchmark_problem: BenchmarkProblem):
    print("Starting run_deap_for_benchmark_problem")
    with announce("Generating the pRef"):
        pRef = benchmark_problem.get_reference_population(sample_size=10000)
    metrics = [Simplicity(), MeanFitness(), Atomicity()]
    for metric in metrics:
        metric.set_pRef(pRef)

    creator.create("FitnessMax", base.Fitness, weights=[1.0 for metric in metrics])
    creator.create("DEAPPSIndividual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    def random_cell_value() -> int:
        return random.choice([-1, 0, 1])


    def ps_to_deap_individual(ps: PS):
        return creator.DEAPPSIndividual(list(ps.values))

    def random_ps_values():
        return ps_to_deap_individual(PS.random(benchmark_problem.search_space))

    def always_empty_ps():
        return ps_to_deap_individual(PS.empty(benchmark_problem.search_space))

    def geometric_distribution_ps():
        result = PS.empty(benchmark_problem.search_space)
        chance_of_success = 0.79
        while random.random() < chance_of_success:
            var_index = random.randrange(benchmark_problem.search_space.amount_of_parameters)
            value = random.randrange(benchmark_problem.search_space.cardinalities[var_index])
            result = result.with_fixed_value(var_index, value)
        return ps_to_deap_individual(result)



    def random_but_sometimes_empty():
        if random.random() < 0.8:
            return always_empty_ps()
        else:
            return random_ps_values()

    toolbox.register("attr_cell", random_cell_value)
    toolbox.register("ps_individual",
                     geometric_distribution_ps)

    def evaluate(ps_individual) -> tuple:
        ps = PS(ps_individual)
        return tuple(metric.get_single_score(ps) for metric in metrics)

    toolbox.register("mate", tools.cxUniform, indpb=1/benchmark_problem.search_space.amount_of_parameters)
    lower_bounds = [-1 for _ in benchmark_problem.search_space.cardinalities]
    upper_bounds = [card-1 for card in benchmark_problem.search_space.cardinalities]
    toolbox.register("mutate", tools.mutUniformInt, low=lower_bounds, up=upper_bounds, indpb=1/benchmark_problem.search_space.amount_of_parameters)

    toolbox.register("evaluate", evaluate)
    toolbox.register("population", tools.initRepeat, list, toolbox.ps_individual)
    #
    #

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)



    with_nsgaiii = True

    if with_nsgaiii:
        ref_points = tools.uniform_reference_points(len(metrics), 12)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

        pop, logbook = nsgaiii(toolbox=toolbox,
                               mu = 300,
                               cxpb=1,
                               mutpb=0.2,
                               ngen=10,
                               stats=stats)
    else:
        toolbox.register("select", tools.selTournament, tournsize=3)
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
    new_pop = []
    for ind in pop:
        new_e_ps = EvaluatedPS(PS(ind))
        new_e_ps.metric_scores = ind.fitness.values
        new_pop.append(new_e_ps)

    pop = list(set(new_pop))
    pop.sort(key=lambda x: x.metric_scores[-1], reverse=True)

    for ind in pop:
        print(benchmark_problem.repr_ps(ind.ps))
        print(f"Has score {ind.metric_scores}\n")





def comprehensive_search(benchmark_problem: BenchmarkProblem,
                         metric: Metric,
                         sample_size: int,
                         amount = 12,
                         reverse=True):
    all_ps = PS.all_possible(benchmark_problem.search_space)

    pRef = benchmark_problem.get_reference_population(sample_size)
    metric.set_pRef(pRef)

    evaluated_pss = [EvaluatedPS(ps) for ps in all_ps]
    for evaluated_ps in evaluated_pss:
        evaluated_ps.aggregated_score = metric.get_single_normalised_score(evaluated_ps.ps)

    evaluated_pss.sort(reverse=reverse)
    print(f"The best in the search space, according to {metric} are")
    to_show = evaluated_pss[:amount] if amount is not None else evaluated_pss
    for e_ps in to_show:
        print(e_ps)



