#!/usr/bin/env python3


""" This is the source code related to the paper 'Mining Potentially Explanatory Patterns via Partial Solutions',
    The authors of the paper are Giancarlo Catalano (me),
      Sandy Brownlee, David Cairns (Stirling Uni)
      John McCall (RGU)
      Russell Ainsle (BT).


    This code is a minimal implementation of the system I proposed in the paper,
    and I hope you can find the answers to any questions that you have!

    I recommend playing around with:
        - The problem being solved
        - The metrics used to search for PSs (find them in PSMiner.with_default_settings)
        - the sample sizes etc...
"""
import itertools

import numpy as np

import TerminationCriteria
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.Checkerboard import CheckerBoard
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from BenchmarkProblems.UnitaryProblem import UnitaryProblem
from EvaluatedFS import EvaluatedFS
from Explainer import Explainer
from PS import STAR, PS
from PSMetric.Atomicity import Atomicity
from PSMetric.BivariateANOVALinkage import BivariateANOVALinkage
from PSMetric.GlobalPerturbation import UnivariateGlobalPerturbation, BivariateGlobalPerturbation
from PSMetric.Linkage import Linkage
from PSMetric.LocalPerturbation import UnivariateLocalPerturbation, BivariateLocalPerturbation
from PSMetric.Metric import test_different_metrics_for_ps
from PSMiner import PSMiner, measure_T2_success_rate
from PickAndMerge import PickAndMergeSampler
from utils import announce, indent


def show_overall_system(benchmark_problem: BenchmarkProblem):
    """
    This function gives an overview of the system:
        1. Generate a reference population (a PRef)
        2. Generate a PS Catalog using the PS Miner
        3. Sample new solutions from the catalog using Pick & Merge
        4. Explain those new solutions using the catalog

    :param benchmark_problem: a benchmark problem, find more in the BenchmarkProblems directory
    :return: Nothing! Just printing
    """

    print(f"The problem is {benchmark_problem}")

    # 1. Generating the reference population
    pRef_size = 10000
    with announce("Generating Reference Population"):
        pRef = benchmark_problem.get_reference_population(pRef_size)
    pRef.describe_self()

    # 2. Obtaining the PS catalog
    ps_miner = PSMiner.with_default_settings(pRef)
    ps_evaluation_budget = 10000
    termination_criterion = TerminationCriteria.PSEvaluationLimit(ps_evaluation_budget)

    with announce("Running the PS Miner"):
        ps_miner.run(termination_criterion)

    ps_catalog = ps_miner.get_results(20)

    print("The catalog consists of:")
    for item in ps_catalog:
        print("\n")
        print(indent(f"{benchmark_problem.repr_ps(item.ps)}, weight = {item.aggregated_score:.3f}"))

    # 3. Sampling new solutions
    print("\nFrom the catalog we can sample new solutions")
    new_solutions_to_produce = 12
    sampler = PickAndMergeSampler(search_space=benchmark_problem.search_space,
                                  individuals=ps_catalog)

    with announce("Sampling from the PS Catalog using Pick & Merge"):
        sampled_solutions = [sampler.sample() for _ in range(new_solutions_to_produce)]

    evaluated_sampled_solutions = [EvaluatedFS(fs, benchmark_problem.fitness_function(fs)) for fs in sampled_solutions]
    evaluated_sampled_solutions.sort(reverse=True)

    for index, sample in enumerate(evaluated_sampled_solutions):
        print(f"[{index}]")
        print(indent(indent(f"{benchmark_problem.repr_fs(sample.full_solution)}, has fitness {sample.fitness:.2f}")))

    # 4. Explainability, at least locally.
    explainer = Explainer(benchmark_problem, ps_catalog, pRef)
    explainer.explanation_loop(evaluated_sampled_solutions)

    print("And that concludes the showcase")




def test_atomicities(benchmark_problem: UnitaryProblem):
    atomicities = [Atomicity(), Linkage(), BivariateANOVALinkage(),
                   UnivariateLocalPerturbation(), BivariateLocalPerturbation(),
                   UnivariateGlobalPerturbation(), BivariateGlobalPerturbation()]

    with announce("Generating the pRef"):
        pRef = benchmark_problem.get_reference_population(10000)

    with announce("Setting the PRefs"):
        for metric in atomicities:
            metric.set_pRef(pRef)

    cliques_to_be_tested: list[np.ndarray]
    clique_size = benchmark_problem.clique_size

    unfixed = PS.empty(benchmark_problem.search_space)
    single_zero = unfixed.with_fixed_value(0, 0)
    single_one = unfixed.with_fixed_value(0, 1)
    pair_zero = single_zero.with_fixed_value(1, 0)
    pair_one = single_one.with_fixed_value(1, 1)
    clique_zero = unfixed.copy()
    clique_zero.values[0:clique_size] = 0
    clique_one = unfixed.copy()
    clique_one.values[0:clique_size] = 1


    test_ps_set = [] #[single_zero, single_one, pair_zero, pair_one, clique_zero, clique_one]

    digits = [STAR, 0, 1]
    for values in itertools.combinations_with_replacement(digits, r=clique_size):
        new_ps = unfixed.copy()
        new_ps.values[0:clique_size] = values
        if not new_ps.is_empty():
            test_ps_set.append(new_ps)

    for test_ps in test_ps_set:
        test_different_metrics_for_ps(test_ps, atomicities)
        print("-"*40)



if __name__ == '__main__':
    # problem = GraphColouring.random(amount_of_nodes=6, amount_of_colours=3, chance_of_connection=0.3)
    # problem = CheckerBoard(4, 4)
    problem = RoyalRoad(5, 4)
    show_overall_system(problem)
