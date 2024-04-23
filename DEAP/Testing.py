import random
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, creator, base, tools
from deap.tools import selNSGA2

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from DEAP.CustomCrowdingMechanism import gc_selNSGA2
from EvaluatedPS import EvaluatedPS
from GA.HistoryPRefs import uniformly_random_distribution_pRef, pRef_from_GA, pRef_from_SA
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric
from PSMetric.Simplicity import Simplicity
from utils import announce


def nsgaii(toolbox,
           stats,
           mu,
           ngen,
           cxpb,
           mutpb):
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "avg", "max"

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
        pop = list(set(pop))
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



def nsgaiii_pure_functionality(toolbox, mu, ngen, cxpb, mutpb):
    def fill_evaluation_gaps(input_pop):
        for ind in input_pop:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        return input_pop

    pop = fill_evaluation_gaps(toolbox.population(n=mu))

    # Begin the generational process
    for gen in range(1, ngen):
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        offspring = fill_evaluation_gaps(offspring)
        pop = toolbox.select(pop + offspring, mu)
    return pop

def get_toolbox_for_problem(benchmark_problem: BenchmarkProblem,
                            metrics: list[Metric],
                            use_experimental_niching = False):
    with announce("Generating the pRef"):
        pRef = benchmark_problem.get_reference_population(sample_size=10000)
    for metric in metrics:
        metric.set_pRef(pRef)

    creator.create("FitnessMax", base.Fitness, weights=[1.0 for metric in metrics])
    creator.create("DEAPPSIndividual", PS,
                   fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    def geometric_distribution_ps():
        result = PS.empty(benchmark_problem.search_space)
        chance_of_success = 0.79
        while random.random() < chance_of_success:
            var_index = random.randrange(benchmark_problem.search_space.amount_of_parameters)
            value = random.randrange(benchmark_problem.search_space.cardinalities[var_index])
            result = result.with_fixed_value(var_index, value)
        return creator.DEAPPSIndividual(result)



    toolbox.register("make_random_ps",
                     geometric_distribution_ps)


    def evaluate(ps) -> tuple:
        return tuple(metric.get_single_score(ps) for metric in metrics)

    toolbox.register("mate", tools.cxUniform, indpb=1/benchmark_problem.search_space.amount_of_parameters)
    lower_bounds = [-1 for _ in benchmark_problem.search_space.cardinalities]
    upper_bounds = [card-1 for card in benchmark_problem.search_space.cardinalities]
    toolbox.register("mutate", tools.mutUniformInt, low=lower_bounds, up=upper_bounds, indpb=1/benchmark_problem.search_space.amount_of_parameters)

    toolbox.register("evaluate", evaluate)
    toolbox.register("population", tools.initRepeat, list, toolbox.make_random_ps)

    if use_experimental_niching:
        toolbox.register("select", gc_selNSGA2)
    else:
        toolbox.register("select", selNSGA2)
    return toolbox

def get_stats_object():
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    return stats


def report_in_order_of_last_metric(population,
                                   benchmark_problem: BenchmarkProblem,
                                   limit_to = None):
    new_pop = []
    for ind in population:
        new_e_ps = EvaluatedPS(PS(ind))
        new_e_ps.metric_scores = ind.fitness.values
        new_pop.append(new_e_ps)

    pop = list(set(new_pop))
    pop.sort(key=lambda x: x.metric_scores[-1], reverse=True)

    amount_to_show = len(pop)
    if limit_to is not None:
        amount_to_show = limit_to
    for ind in pop[:amount_to_show]:
        print(benchmark_problem.repr_ps(ind.ps))
        print(f"Has score {ind.metric_scores}\n")


def plot_stats_for_run(logbook,
                       metrics: list[Metric],
                       show_max = False,
                       show_mean = True):
    generations = logbook.select("gen")
    num_variables = len(metrics)


    avg_matrix = np.array([logbook[generation]["avg"] for generation in generations])
    max_matrix = np.array([logbook[generation]["max"] for generation in generations])

    # Create a new figure with subplots
    fig, axs = plt.subplots(1, num_variables, figsize=(12, 6))  # 1 row, `num_variables` columns

    # Loop through each variable to create a subplot
    for metric_index, metric in enumerate(metrics):

        if show_mean:
            axs[metric_index].plot(generations, avg_matrix[:, metric_index], label='Average', linestyle='-', marker='o')
        if show_max:
            axs[metric_index].plot(generations, max_matrix[:, metric_index], label='Maximum', linestyle='--', marker='x')

        axs[metric_index].set_xlabel('Generation')
        axs[metric_index].set_ylabel('Value')
        axs[metric_index].set_title(f'{metric}')
        axs[metric_index].legend()  # Add legend to each subplot

    # Adjust layout to ensure plots don't overlap
    plt.tight_layout()

    # Display the plots
    plt.show()
def run_deap_for_benchmark_problem(benchmark_problem: BenchmarkProblem):
    print("Starting run_deap_for_benchmark_problem")
    metrics = [Simplicity(), MeanFitness(), Atomicity()]
    with announce("Running the algorithm"):
        pop, logbook = nsgaii(toolbox=get_toolbox_for_problem(benchmark_problem, metrics, use_experimental_niching=True),
                              mu = 300,
                              cxpb=0.5,
                              mutpb=1/benchmark_problem.search_space.amount_of_parameters,
                              ngen=100,
                              stats=get_stats_object())

    print("The last population is ")
    report_in_order_of_last_metric(pop, benchmark_problem)

    plot_stats_for_run(logbook, metrics)








def comprehensive_search(benchmark_problem: BenchmarkProblem,
                         metric: Metric,
                         sample_size: int,
                         amount = 12,
                         reverse=True):
    all_ps = PS.all_possible(benchmark_problem.search_space)

    with announce("Generating the PRef"):
        pRef = benchmark_problem.get_reference_population(sample_size)
    metric.set_pRef(pRef)

    evaluated_pss = [EvaluatedPS(ps) for ps in all_ps]
    #evaluated_pss = [ps for ps in evaluated_pss if ps.ps.fixed_count() < 7]
    with announce(f"Evaluating all the PSs, there are {len(evaluated_pss)} of them"):
        for evaluated_ps in evaluated_pss:
            evaluated_ps.aggregated_score = metric.get_single_normalised_score(evaluated_ps.ps)

    evaluated_pss.sort(reverse=reverse)
    print(f"The best in the search space, according to {metric} are")
    to_show = evaluated_pss[:amount] if amount is not None else evaluated_pss
    for e_ps in to_show:
        print(e_ps)



def run_nsgaii_on_history_pRef(benchmark_problem: BenchmarkProblem,
                               which_algorithm: Literal["uniform", "GA", "SA"],
                               ps_miner_generations = 100,
                               sample_size= 10000):
    pRef = None
    match which_algorithm:
        case "uniform": pRef = uniformly_random_distribution_pRef(sample_size=sample_size,
                                                                  benchmark_problem=benchmark_problem)
        case "GA": pRef = pRef_from_GA(benchmark_problem=benchmark_problem,
                                       sample_size=sample_size,
                                       ga_population_size=300)
        case "SA": pRef = pRef_from_SA(benchmark_problem=benchmark_problem,
                                       sample_size=sample_size,
                                       max_trace = sample_size)
        case _: raise ValueError


    metrics = [Simplicity(), MeanFitness(), Atomicity()]
    with announce("Running the PS_mining algorithm"):
        toolbox = get_toolbox_for_problem(benchmark_problem,
                                          metrics, use_experimental_niching=True)
        final_population, logbook = nsgaii(toolbox=toolbox,
                              mu = 300,
                              cxpb=0.5,
                              mutpb=1/benchmark_problem.search_space.amount_of_parameters,
                              ngen=ps_miner_generations,
                              stats=get_stats_object())

    print("The last population is ")
    report_in_order_of_last_metric(final_population,
                                   benchmark_problem,
                                   limit_to = 12)
    print("And now plotting the stats!")
    plot_stats_for_run(logbook, metrics)


