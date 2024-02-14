import random
from typing import Callable

import TerminationCriteria
from BaselineApproaches.Evaluator import PSEvaluator
from BaselineApproaches.FullSolutionGA import FullSolutionGA
from BaselineApproaches.PSGA import PSGA
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.BinVal import BinVal
from BenchmarkProblems.OneMax import OneMax
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.RoyalRoadWithOverlaps import RoyalRoadWithOverlaps
from BenchmarkProblems.Trapk import Trapk
from FullSolution import FullSolution
from JMetal.PSProblem import test_PSProblem
from JMetal.TestProblem import test_JMetal_integer
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.KindaAtomicity import LinkageViaMeanFitDiff, SimplerAtomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from PSMetric.Opposite import Opposite
from PSMiners.ArchiveBasedSpecialisationMiner import ABSM
from PSMiners.ModifiedArchiveMiner import ModifiedArchiveMiner
from PSMiners.SelfAssembly import SelfAssembly
from PSMiners.SemelpariousMiner import SemelpariousMiner
from SearchSpace import SearchSpace
from custom_types import Fitness


def test_archive_miner(problem: BenchmarkProblem):
    pRef: PRef = problem.get_pRef(10000)

    simplicity = Simplicity()
    atomicity = Atomicity()
    meanFitness = MeanFitness()

    ps_evaluator = PSEvaluator([simplicity, atomicity, meanFitness], pRef)

    budget_limit = TerminationCriteria.EvaluationBudgetLimit(10000)
    #iteration_limit = TerminationCriteria.IterationLimit(12)
    #termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, iteration_limit)

    miner = ABSM(150, ps_evaluator)

    miner.run(budget_limit)

    results = miner.get_results(quantity_returned=10)
    print("The results of the PSABSM are:")
    for ps, fitness in results:
        print(f"PS: {ps}, fitness = {fitness}")

    print(f"The used budget is {miner.ps_evaluator.used_evaluations}")


def test_modified_archive_miner(problem: BenchmarkProblem):
    print("Testing the modified archive miner")
    pRef: PRef = problem.get_pRef(10000)

    budget_limit = TerminationCriteria.EvaluationBudgetLimit(10000)
    # iteration_limit = TerminationCriteria.IterationLimit(12)
    # termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, iteration_limit)

    miner = ModifiedArchiveMiner(150, pRef)

    miner.run(budget_limit)

    results = miner.get_results(quantity_returned=10)
    print("The results of the PSABSM are:")
    for ps, fitness in results:
        print(f"PS: {ps}, fitness = {fitness}")

    print(f"The used budget is {miner.evaluator.evaluations}")



def compare_own_miners(problem: BenchmarkProblem):
    print(f"The problem is {problem.long_repr()}")
    pRef: PRef = problem.get_pRef(10000)
    budget_limit = TerminationCriteria.EvaluationBudgetLimit(10000)

    ps_evaluator = PSEvaluator([Simplicity(), MeanFitness(), Atomicity()], pRef)
    traditional_miner = ABSM(150, ps_evaluator)

    experimental_miner = ModifiedArchiveMiner(150, pRef)

    def run_and_show_results(miner):
        miner.run(budget_limit)
        returned_from_experimental = miner.get_results(10)
        for ps, fitness in returned_from_experimental:
            print(f"PS: {ps}, fitness = {fitness}")
        print(f"The used budget is {miner.get_used_evaluations()}")


    print("Running the experimental miner")
    run_and_show_results(experimental_miner)

    print("Running the traditional miner")
    run_and_show_results(traditional_miner)




def test_MO(problem: BenchmarkProblem):
    algorithms = ["NSGAII", "MOEAD", "MOCell", "GDE3"]

    for algorithm in algorithms:
        print(f"\n\nTesting with {algorithm}")
        test_PSProblem(problem, which=algorithm)




def test_semelparious_miner(problem: BenchmarkProblem):
    pRef = problem.get_pRef(10000)
    spm = SemelpariousMiner(population_size=150, pRef=pRef)
    budget_limit = TerminationCriteria.EvaluationBudgetLimit(10000)
    spm.run(budget_limit)

    results = spm.get_results(0)
    print("The results of the SemelpariousMiner are:")
    for individual in results:
        ps = individual.ps
        simplicity, mean_fitness, atomicity = individual.metrics
        print(f"PS: {ps}, "
              f"\t{simplicity:.0f},"
              f"\t{mean_fitness:.2f},"
              f"\t{atomicity:.4f}")

    print(f"The used budget is {spm.evaluations}")


def test_simple_hill_climber(problem: BenchmarkProblem):
    sa = SelfAssembly(problem.get_pRef(10000))

    for _ in range(60):
        if random.random() < 0.5:
            starting_point = sa.select_random_fixed_start()
        else:
            starting_point = sa.unfixed_start()

        final = sa.improve_continously(starting_point)

        print(f"Starting from {starting_point}, we got to {final}")


    print("At the end, the top 12 best partial solutions are")
    for ps, count in sa.ps_counter.most_common(12):
        print(f"{ps}, appearing {count} times")


def print_separator():
    print("-"*30)




def test_many_miners():
    problem = RoyalRoad(4, 5)
    # print(f"The problem is {problem.long_repr()}")
    print_separator()
    test_archive_miner(problem)
    print_separator()
    test_MO(problem)
    print_separator()
    test_simple_hill_climber(problem)






if __name__ == '__main__':
    problem = RoyalRoadWithOverlaps(3, 5, amount_of_bits=12)
    compare_own_miners(problem)





