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
from JMetal.TestProblem import test_JMetal
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.KindaAtomicity import LinkageViaMeanFitDiff, SimplerAtomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from PSMetric.Opposite import Opposite
from PSMiners.ArchiveBasedSpecialisationMiner import ABSM
from SearchSpace import SearchSpace
from custom_types import Fitness


def test_fsga(problem: BenchmarkProblem):
    budget_limit = TerminationCriteria.EvaluationBudgetLimit(10000)
    found_global = TerminationCriteria.UntilGlobalOptimaReached(3)
    termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, found_global)

    fs_ga = FullSolutionGA(search_space=problem.search_space,
                           crossover_rate=0.5,
                           mutation_rate=0.1,
                           elite_size=2,
                           tournament_size=3,
                           population_size=100,
                           fitness_function=problem.fitness_function)

    fs_ga.run(termination_criteria)

    results = fs_ga.get_best_of_last_run(quantity_returned=3)
    print("The results of the FSGA are:")
    for fs, fitness in results:
        print(f"FS: {fs}, fitness = {fitness}")


def test_psabsm(problem: BenchmarkProblem):
    pRef: PRef = problem.get_pRef(1000)

    simplicity = Simplicity()
    mean_fitness = MeanFitness()
    atomicity = LinkageViaMeanFitDiff()

    ps_evaluator = PSEvaluator([simplicity, mean_fitness, atomicity], pRef)

    budget_limit = TerminationCriteria.EvaluationBudgetLimit(10000)
    iteration_limit = TerminationCriteria.IterationLimit(12)
    termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, iteration_limit)

    absm = ABSM(150, ps_evaluator)

    absm.run(termination_criteria)

    results = absm.get_best_of_last_run(quantity_returned=10)
    print("The results of the PSABSM are:")
    for ps, fitness in results:
        print(f"PS: {ps}, fitness = {fitness}")

    print(f"The used budget is {absm.ps_evaluator.used_evaluations}")



def test_atomicity(problem: BenchmarkProblem):
    pRef: PRef = problem.get_pRef(10000)

    atomicity = LinkageViaMeanFitDiff()
    ps_evaluator = PSEvaluator([atomicity], pRef)
    sample_pss = [PS.random(problem.search_space) for _ in range(6)]

    scores = ps_evaluator.evaluate_population_with_raw_scores(sample_pss)
    print(f"The scores are")
    for ps, score in scores:
        print(f"{ps}, score = {score}")

def test_simplicity_and_atomicity(problem: BenchmarkProblem):
    pRef: PRef = problem.get_pRef(10000)

    complexity = Opposite(Simplicity())
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


def test_atomicity_2(problem: RoyalRoadWithOverlaps):
    pRef: PRef = problem.get_pRef(10000)

    target_a, target_b = None, None

    choices = list(problem.target_pss)
    while True:
        random_a = random.choice(choices)
        random_b = random.choice(choices)
        if PS.mergeable(random_a, random_b) and random_a!=random_b:
            target_a, target_b = random_a, random_b
            break

    united = PS.merge(target_a, target_b)
    sample_pss = [target_a, target_b, united]


    atomicity = LinkageViaMeanFitDiff()
    ps_evaluator = PSEvaluator([atomicity], pRef)

    scores = ps_evaluator.evaluate_population_with_raw_scores(sample_pss)
    print(f"The scores are")
    for ps, score in scores:
        print(f"{ps}, score = {score}")



if __name__ == '__main__':
    # problem = Trapk(3, 5)
    # print(f"The problem is {problem.long_repr()}")
    # test_simplicity_and_atomicity(problem)

    test_JMetal()
