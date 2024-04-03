#!/usr/bin/env python3
import TerminationCriteria
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.Checkerboard import CheckerBoard
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from EvaluatedFS import EvaluatedFS
from Explainer import Explainer
from GCArchiveMiner import GCArchiveMiner
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
        print(f"[{index}]\t\t{sample.full_solution}, has fitness {sample.fitness:.2f}")


    explainer = Explainer(benchmark_problem, ps_catalog, pRef)

    first_round = True

    while True:
        if first_round:
            print("Would you like to see some explanations of the solutions? Write an index, or n to exit")
        else:
            print("Type another index, or n to exit")
        answer = input()
        if answer.upper() == "N":
            break
        else:
            try:
                index = int(answer)
            except:
                print("That didn't work, please retry")
                continue
            solution_to_explain = evaluated_sampled_solutions[index]
            explainer.local_explanation_of_full_solution(solution_to_explain.full_solution)

    print("And that concludes the showcase")




if __name__ == '__main__':
    problem = CheckerBoard(3, 3)
    show_overall_system(problem)
