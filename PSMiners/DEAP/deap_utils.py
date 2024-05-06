import random
from typing import Literal, Any

import deap.base
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, creator, base, tools
from deap.tools import selNSGA2, uniform_reference_points, selNSGA3WithMemory

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef
from Core.PSMiner import PSMiner
from Core.TerminationCriteria import TerminationCriteria
from PSMiners.AbstractPSMiner import AbstractPSMiner
from PSMiners.DEAP.CustomCrowdingMechanism import gc_selNSGA2, GC_selNSGA3WithMemory
from FSStochasticSearch.HistoryPRefs import uniformly_random_distribution_pRef, pRef_from_GA, pRef_from_SA
from Core.PS import PS
from Core.PSMetric.Atomicity import Atomicity
from Core.PSMetric.Classic3 import Classic3PSMetrics
from Core.PSMetric.MeanFitness import MeanFitness
from Core.PSMetric.Metric import Metric
from Core.PSMetric.Simplicity import Simplicity
from utils import announce


def nsga(toolbox,
         stats,
         mu,
         termination_criteria: TerminationCriteria,
         cxpb,
         mutpb,
         classic3_evaluator: Classic3PSMetrics,
         verbose=False):
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
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    iterations = 0
    def should_stop():
        return termination_criteria.met(ps_evaluations = classic3_evaluator.used_evaluations, iterations=iterations)

    while not should_stop():
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
        logbook.record(gen=iterations, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        iterations +=1

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

def get_toolbox_for_problem(pRef: PRef,
                            classic3_evaluator: Classic3PSMetrics,
                            uses_experimental_crowding = True):
    creator.create("FitnessMax", base.Fitness, weights=[1.0, 1.0, 1.0])
    creator.create("DEAPPSIndividual", PS,
                   fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    search_space = pRef.search_space
    def geometric_distribution_ps():
        result = PS.empty(search_space)
        chance_of_success = 0.79
        while random.random() < chance_of_success:
            var_index = random.randrange(search_space.amount_of_parameters)
            value = random.randrange(search_space.cardinalities[var_index])
            result = result.with_fixed_value(var_index, value)
        return creator.DEAPPSIndividual(result)



    toolbox.register("make_random_ps",
                     geometric_distribution_ps)
    def evaluate(ps) -> tuple:
        return classic3_evaluator.get_S_MF_A(ps)  # experimental

    toolbox.register("mate", tools.cxUniform, indpb=1/search_space.amount_of_parameters)
    lower_bounds = [-1 for _ in search_space.cardinalities]
    upper_bounds = [card-1 for card in search_space.cardinalities]
    toolbox.register("mutate", tools.mutUniformInt, low=lower_bounds, up=upper_bounds, indpb=1/search_space.amount_of_parameters)

    toolbox.register("evaluate", evaluate)
    toolbox.register("population", tools.initRepeat, list, toolbox.make_random_ps)

    selection_method = None


    ref_points = uniform_reference_points(nobj=3, p=12)
    selection_method = GC_selNSGA3WithMemory(ref_points) if uses_experimental_crowding else selNSGA3WithMemory(ref_points)
    toolbox.register("select", selection_method)
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
                       figure_name: str,
                       show_max = False,
                       show_mean = True,
                       ):
    generations = logbook.select("gen")
    metric_labels = ["Simplicity", "MeanFitness", "Atomicity"]
    num_variables = len(metric_labels)


    avg_matrix = np.array([logbook[generation]["avg"] for generation in generations])
    max_matrix = np.array([logbook[generation]["max"] for generation in generations])

    # Create a new figure with subplots
    fig, axs = plt.subplots(1, num_variables, figsize=(12, 6))  # 1 row, `num_variables` columns

    # Loop through each variable to create a subplot
    for metric_index, metric_label in enumerate(metric_labels):

        if show_mean:
            axs[metric_index].plot(generations, avg_matrix[:, metric_index], label='Average', linestyle='-', marker='o')
        if show_max:
            axs[metric_index].plot(generations, max_matrix[:, metric_index], label='Maximum', linestyle='--', marker='x')

        axs[metric_index].set_xlabel('Generation')
        axs[metric_index].set_ylabel('Value')
        axs[metric_index].set_title(metric_label)
        axs[metric_index].legend()  # Add legend to each subplot

    # Adjust layout to ensure plots don't overlap
    plt.tight_layout()

    # Display the plots
    plt.savefig(figure_name)


def comprehensive_search(benchmark_problem: BenchmarkProblem,
                         metric: Metric,
                         sample_size: int,
                         amount = 12,
                         reverse=True):
    """This is a debug function, to check what the global optimum of an objective is"""
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








