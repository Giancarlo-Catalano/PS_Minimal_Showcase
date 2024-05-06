from typing import Literal

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core import TerminationCriteria
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef
from Core.PSMiner import PSMiner
from FSStochasticSearch.HistoryPRefs import uniformly_random_distribution_pRef, pRef_from_GA, pRef_from_SA, \
    pRef_from_GA_best, pRef_from_SA_best
from PSMiners.DEAP.NSGAPSMiner import NSGAPSMiner
from utils import announce


def get_history_pRef(benchmark_problem: BenchmarkProblem,
                     sample_size: int,
                     which_algorithm: Literal["uniform", "GA", "SA", "GA_best", "SA_best"],
                     verbose=True):
    with announce(f"Running the algorithm to generate the PRef using {which_algorithm}", verbose=verbose):
        match which_algorithm:
            case "uniform": return uniformly_random_distribution_pRef(sample_size=sample_size,
                                                                      benchmark_problem=benchmark_problem)
            case "GA": return pRef_from_GA(benchmark_problem=benchmark_problem,
                                           sample_size=sample_size,
                                           ga_population_size=300)
            case "SA": return pRef_from_SA(benchmark_problem=benchmark_problem,
                                           sample_size=sample_size,
                                           max_trace = sample_size)
            case "GA_best": return pRef_from_GA_best(benchmark_problem=benchmark_problem,
                                                     sample_size=sample_size,
                                                     fs_evaluation_budget=sample_size * 100, # TODO decide elsewhere
                                                     )
            case "SA_best": return pRef_from_SA_best(benchmark_problem=benchmark_problem,
                                                     sample_size=sample_size)
            case _: raise ValueError






def mine_with_old_method(pRef: PRef,
                         ps_budget: int) -> list[EvaluatedPS]:
    ps_miner = PSMiner.with_default_settings(pRef)
    termination_criterion = TerminationCriteria.PSEvaluationLimit(ps_budget)

    with announce("Running the Core Miner"):
        ps_miner.run(termination_criterion)

    return ps_miner.get_results(ps_miner.population_size)


def mine_with_new_method(pRef: PRef,
                         ps_budget: int) -> list[EvaluatedPS]:
    ps_miner = NSGAPSMiner.with_default_settings(pRef)
    termination_criterion = TerminationCriteria.PSEvaluationLimit(ps_budget)

    with announce("Running the Core Miner"):
        ps_miner.run(termination_criterion)

    return ps_miner.get_results(ps_miner.population_size)



