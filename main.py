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
import json
import os

import numpy as np

import TerminationCriteria
import utils
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.Checkerboard import CheckerBoard
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from BenchmarkProblems.UnitaryProblem import UnitaryProblem
from DEAP.Testing import run_deap_for_benchmark_problem, comprehensive_search, run_nsgaii_on_history_pRef
from EvaluatedFS import EvaluatedFS
from Experimentation.DetectingPatterns import test_and_produce_patterns, plot_nicely, json_to_cohorts, cohorts_to_json, \
    BTProblemPatternDetector, mine_cohorts_from_problem, analyse_data_from_json_cohorts, \
    generate_control_data_for_cohorts, generate_coverage_stats
from Explainer import Explainer
from PS import STAR, PS
from PSMetric.Atomicity import Atomicity
from PSMetric.BivariateANOVALinkage import BivariateANOVALinkage
from PSMetric.Classic3 import test_classic3
from PSMetric.GlobalPerturbation import UnivariateGlobalPerturbation, BivariateGlobalPerturbation
from PSMetric.Linkage import Linkage
from PSMetric.LocalPerturbation import UnivariateLocalPerturbation, BivariateLocalPerturbation
from PSMetric.Metric import test_different_metrics_for_ps
from PSMiner import PSMiner, measure_T2_success_rate
from PSMiners.MuPlusLambda.MPLLR import MPLLR
from PickAndMerge import PickAndMergeSampler
from pymootesting import test_run_with_pymoo
from utils import announce, indent

from pdf.testing import demo_hello_world


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



def mine_cohorts_and_write_to_file(benchmark_problem: BenchmarkProblem,
                                   output_file_name: str,
                                   verbose=True):

    if verbose:
        print("Initialising mine_cohorts_and_write_to_file(")
        print(f"\tbenchmark_problem={benchmark_problem},")
        print(f"\toutput_file_name={output_file_name}")

    cohorts = mine_cohorts_from_problem(benchmark_problem=problem,
                                        method="SA",
                                        pRef_size=1000,
                                        nsga_pop_size=600,
                                        nsga_ngens=600,
                                        verbose=verbose)

    with announce(f"Writing the cohorts ({len(cohorts)} onto the file", verbose):
        data = cohorts_to_json(cohorts)
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
        with open(output_file_name, "w+") as file:
            json.dump(data, file)

    if verbose:
        print(f"Finished writing onto {output_file_name}")

def analyse_cohort_data(benchmark_problem: BTProblem,
                        cohorts_json_file_name: str,
                        csv_file_name: str,
                        verbose=True):
    with announce(f"Reading the file {cohorts_json_file_name} to obtain the cohorts", verbose):
        cohorts = json_to_cohorts(cohorts_json_file_name)


    detector = BTProblemPatternDetector(benchmark_problem)
    control_cohorts = detector.generate_matching_random_cohorts(cohorts,
                                                                amount_to_generate=len(cohorts))


    analyse_data_from_json_cohorts(problem = problem,
                                   real_cohorts=cohorts,
                                   control_cohorts=control_cohorts,
                                   output_csv_file_name = csv_file_name,
                                   verbose=verbose)


if __name__ == '__main__':
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation"
    current_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\cohorts_17'02_29-04"
    # current_directory = os.path.join(experimental_directory, "cohorts_"+utils.get_formatted_timestamp())
    cohort_file = os.path.join(current_directory, "cohort.json")
    csv_file = os.path.join(current_directory, "analysis.csv")

    problem = EfficientBTProblem.from_default_files()
    #problem = Trapk(4, 4)
    mine_cohorts_and_write_to_file(problem, cohort_file, verbose=True)

    #analyse_cohort_data(problem, cohort_file, csv_file, True)

    # cohorts = json_to_cohorts(cohort_file)
    # coverage = generate_coverage_stats(problem, cohorts)
    #
    # for worker_id in coverage:
    #     print(f"{worker_id}\t{coverage[worker_id]}")