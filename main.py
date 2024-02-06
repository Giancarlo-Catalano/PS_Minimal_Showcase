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
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.KindaAtomicity import KindaAtomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
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

    results = fs_ga.get_best_of_last_run(amount=3)
    print("The results of the FSGA are:")
    for fs, fitness in results:
        print(f"FS: {fs}, fitness = {fitness}")


def test_psabsm(problem: BenchmarkProblem):
    pRef: PRef = problem.get_pRef(1000)

    simplicity = Simplicity()
    mean_fitness = MeanFitness()
    atomicity = Atomicity()

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


if __name__ == '__main__':
    problem = RoyalRoadWithOverlaps(2, 4, 7)
    print(f"The problem is {problem.long_repr()}")
    test_fsga(problem)
    test_psabsm(problem)