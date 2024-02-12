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

    results = miner.get_best_of_last_run(quantity_returned=10)
    print("The results of the PSABSM are:")
    for ps, fitness in results:
        print(f"PS: {ps}, fitness = {fitness}")

    print(f"The used budget is {miner.ps_evaluator.used_evaluations}")


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





if __name__ == '__main__':
    problem = RoyalRoad(2, 5)
    # print(f"The problem is {problem.long_repr()}")
    # test_simplicity_and_atomicity(problem)
    #test_MO(problem)

    test_semelparious_miner(problem)





