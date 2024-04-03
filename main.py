#!/usr/bin/env python3
import TerminationCriteria
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.Checkerboard import CheckerBoard
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.ParityProblem import ParityProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from EvaluatedFS import EvaluatedFS
from Explainer import Explainer
from PSMiner import GCArchiveMiner
from PickAndMerge import PickAndMergeSampler
from utils import announce, indent


def show_overall_system(benchmark_problem: BenchmarkProblem):
    print(f"The problem is {benchmark_problem}")
    pRef_size = 10000
    with announce("Generating Reference Population"):
        pRef = benchmark_problem.get_reference_population(pRef_size)

    pRef.describe_self()

    ps_miner = GCArchiveMiner.with_default_settings(pRef)
    ps_evaluation_budget = 10000
    termination_criterion = TerminationCriteria.PSEvaluationLimit(ps_evaluation_budget)

    with announce("Running the PS Miner"):
        ps_miner.run(termination_criterion)

    ps_catalog = ps_miner.get_results(20)

    print("The catalog consists of:")
    for item in ps_catalog:
        print("\n")
        print(indent(f"{benchmark_problem.repr_ps(item.ps)}, weight = {item.aggregated_score:.3f}"))

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


    explainer = Explainer(benchmark_problem, ps_catalog, pRef)
    explainer.explanation_loop(evaluated_sampled_solutions)

    print("And that concludes the showcase")




if __name__ == '__main__':
    #problem = GraphColouring.random(amount_of_nodes=6, amount_of_colours=3, chance_of_connection=0.3)
    problem = CheckerBoard(4,4)
    #problem = RoyalRoad(4, 4)
    show_overall_system(problem)
