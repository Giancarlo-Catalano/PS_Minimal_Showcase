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
import csv
import json
import os
from typing import Optional

import utils
from BenchmarkProblems.Checkerboard import CheckerBoard
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.MultiDimensionalKnapsack import MultiDimensionalKnapsack
from Core import TerminationCriteria
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.EvaluatedPS import EvaluatedPS
from Core.PS import PS
from Core.Explainer import Explainer
from Core.ArchivePSMiner import ArchivePSMiner
from Core.PickAndMerge import PickAndMergeSampler
from Explanation.BT.BTDetector import BTDetector
from Explanation.GC.GCDetector import GCDetector
from FSStochasticSearch.Operators import SinglePointFSMutation
from FSStochasticSearch.SA import SA
from PSMiners.DEAP.NSGAPSMiner import NSGAPSMiner
from PSMiners.DEAP.deap_utils import plot_stats_for_run, report_in_order_of_last_metric
from PSMiners.Mining import get_history_pRef, obtain_pss
from utils import announce, indent


def show_overall_system(benchmark_problem: BenchmarkProblem):
    """
    This function gives an overview of the system:
        1. Generate a reference population (a PRef)
        2. Generate a Core Catalog using the Core Miner
        3. Sample new solutions from the catalog using Pick & Merge
        4. Explain those new solutions using the catalog

    :param benchmark_problem: a benchmark problem, find more in the BenchmarkProblems directory
    :return: Nothing! Just printing
    """

    print(f"The problem is {benchmark_problem}")

    # 1. Generating the reference population
    pRef_size = 10000
    with announce("Generating Reference Population"):
        pRef = get_history_pRef(benchmark_problem, sample_size=pRef_size, which_algorithm="SA")
    pRef.describe_self()

    # 2. Obtaining the Core catalog
    ps_miner = NSGAPSMiner.with_default_settings(pRef)
    ps_evaluation_budget = 10000
    termination_criterion = TerminationCriteria.PSEvaluationLimit(ps_evaluation_budget)

    with announce("Running the PS Miner"):
        ps_miner.run(termination_criterion, verbose=True)

    ps_catalog = ps_miner.get_results(None)
    ps_catalog = list(set(ps_catalog))
    ps_catalog = [item for item in ps_catalog if not item.is_empty()]

    print("The catalog consists of:")
    for item in ps_catalog:
        print("\n")
        print(indent(f"{benchmark_problem.repr_ps(item)}"))

    # 3. Sampling new solutions
    print("\nFrom the catalog we can sample new solutions")
    new_solutions_to_produce = 12
    sampler = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   cooling_coefficient=0.9995)

    solutions = pRef.get_evaluated_FSs()
    solutions = list(set(solutions))
    solutions.sort(reverse=True)


    for index, sample in enumerate(solutions[:6]):
        print(f"[{index}]")
        print(indent(indent(f"{benchmark_problem.repr_fs(sample.full_solution)}, has fitness {sample.fitness:.2f}")))

    # 4. Explainability, at least locally.
    explainer = Explainer(benchmark_problem, ps_catalog, pRef)
    explainer.explanation_loop(solutions)

    print("And that concludes the showcase")

def explanation_loop_for_bt():
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BTDetector"
    problem = EfficientBTProblem.from_default_files()
    detector = BTDetector(problem=problem,
                          folder=experimental_directory,
                          speciality_threshold=0.25,
                          verbose=True)

    # only run this on the first run
    # detector.generate_files_with_default_settings()
    detector.explanation_loop(amount_of_fs_to_propose=6, ps_show_limit=12)


def explanation_loop_for_gc():
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\GCDetector"
    problem_file = os.path.join(experimental_directory, "islets.json")
    problem = GraphColouring.from_file(problem_file)
    problem.view()
    detector = GCDetector(folder = experimental_directory,
                          problem = problem,
                          speciality_threshold=0.25,
                          verbose=True)
    # only run this on the first run
    #detector.generate_files_with_default_settings()
    detector.explanation_loop(amount_of_fs_to_propose=6, ps_show_limit=12)


if __name__ == '__main__':
    explanation_loop_for_bt()
