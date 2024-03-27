import heapq
import json
import logging
import sys
import warnings

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.RoyalRoadWithOverlaps import RoyalRoadWithOverlaps
from BenchmarkProblems.Trapk import Trapk
from EDA.FSIndividual import FSIndividual
from FullSolution import FullSolution
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.Averager import Averager
from PSMetric.BiVariateANOVALinkage import BiVariateANOVALinkage
from PSMetric.Linkage import Linkage
from PSMetric.LocalPerturbation import BivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric, MultipleMetrics
from PSMetric.Simplicity import Simplicity
from PSMiners.Archivers.BaselineArchiveMiner import BaselineArchiveMiner
from PSMiners.Individual import Metrics
from PSMiners.MuPlusLambda.MPLLR import MPLLR
from PSMiners.MuPlusLambda.MuPlusLambda import MuPlusLambda
from PSMiners.Operators.PSMutationOperator import PSMutationOperator, MultimodalMutationOperator, SinglePointMutation
from PSMiners.Operators.PSSelectionOperator import PSSelectionOperator, TruncationSelection, TournamentSelection, \
    AlternatingSelection
from PSMiners.PSMiner import PSMiner, ResultsAsJSON
from TerminationCriteria import PSEvaluationLimit
from utils import execution_time


def test_algorithm(algorithm_instance: PSMiner,
                   evaluation_budget: int,
                   target_PS: list[PS]) -> ResultsAsJSON:
    with execution_time() as timer:
        results_of_run = algorithm_instance.run(termination_criteria=PSEvaluationLimit(evaluation_budget))

    results_of_run["runtime"] = timer.execution_time

    final_population = algorithm_instance.get_results(len(algorithm_instance.current_population))
    final_population: list[PS] = [individual.ps for individual in final_population]

    def index_in_final_population(target: PS) -> int:
        try:
            index = final_population.index(target)
            return index
        except ValueError:
            return -1

    indexes_of_targets = [index_in_final_population(target) for target in target_PS]

    results_of_run["indexes_of_targets"] = indexes_of_targets

    results_of_run["parameters"] = algorithm_instance.get_parameters_as_dict()

    return results_of_run


def get_pRef_from_benchmark_problem(benchmark_problem: BenchmarkProblem, sample_size: int) -> PRef:
    return PRef.sample_from_search_space(search_space=benchmark_problem.search_space,
                                         fitness_function=benchmark_problem.fitness_function,
                                         amount_of_samples=sample_size)


def get_filtered_pRef_from_benchmark_problem(benchmark_problem: BenchmarkProblem,
                                             sample_size: int,
                                             oversampling_factor: int) -> PRef:
    total_samples = (1 + oversampling_factor) * sample_size
    samples = [FullSolution.random(benchmark_problem.search_space) for _ in range(total_samples)]
    individuals = [FSIndividual(sample, benchmark_problem.fitness_function(sample)) for sample in samples]

    best = heapq.nlargest(n=sample_size, iterable=individuals)
    best_solutions = [individual.full_solution for individual in best]
    best_fitnesses = [individual.fitness for individual in best]

    return PRef.from_full_solutions(full_solutions=best_solutions,
                                    fitness_values=best_fitnesses,
                                    search_space=benchmark_problem.search_space)


def get_variants_for_MPL(pRef: PRef,
                         metrics: list[Metric],
                         mutation_operators: list[PSMutationOperator],
                         selection_operators: list[PSSelectionOperator],
                         mus: list[int]) -> list[PSMiner]:
    return [MuPlusLambda(mu_parameter=mu,
                         lambda_parameter=mu * 6,
                         metric=metric,
                         mutation_operator=mutation_operator,
                         selection_operator=selection_operator,
                         pRef=pRef)
            for mu in mus
            for metric in metrics
            for mutation_operator in mutation_operators
            for selection_operator in selection_operators]


def get_variants_for_MPL_lexicase(pRef: PRef,
                                  metrics: list[Metric],
                                  mutation_operators: list[PSMutationOperator],
                                  selection_operators: list[PSSelectionOperator],
                                  mus: list[int]) -> list[PSMiner]:
    return [MuPlusLambda(mu_parameter=mu,
                         lambda_parameter=mu * 6,
                         metric=metric,
                         mutation_operator=mutation_operator,
                         selection_operator=selection_operator,
                         pRef=pRef)
            for mu in mus
            for metric in metrics
            for mutation_operator in mutation_operators
            for selection_operator in selection_operators]


def get_variants_for_MPLLR(pRef: PRef,
                           metrics: list[Metric],
                           mutation_operators: list[PSMutationOperator],
                           selection_operators: list[PSSelectionOperator],
                           mus: list[int],
                           food_weights: list[float]) -> list[PSMiner]:
    return [MPLLR(mu_parameter=mu,
                  lambda_parameter=mu * 6,
                  metric=metric,
                  mutation_operator=mutation_operator,
                  selection_operator=selection_operator,
                  food_weight=food_weight,
                  pRef=pRef)
            for mu in mus
            for food_weight in food_weights
            for mutation_operator in mutation_operators
            for selection_operator in selection_operators
            for metric in metrics]


def get_variants_for_archive_miner(pRef: PRef,
                                   metrics: list[Metric],
                                   population_sizes: list[int]) -> list[PSMiner]:
    return [BaselineArchiveMiner(metric=metric,
                                 pRef=pRef,
                                 population_size=population_size,
                                 offspring_population_size=population_size)
            for metric in metrics
            for population_size in population_sizes]


def get_all_variants_for_miners(pRef: PRef):
    mean_fitness = MeanFitness()
    linkage = Linkage()
    bal = BiVariateANOVALinkage()
    simplicity = Simplicity()
    legacy_atomicity = Atomicity()
    novel_atomicity = BivariateLocalPerturbation()

    # setting the pRef here so that the algorithms don't have to, since it would waste a lot of time to recalculate the linkage every time
    for metric in [mean_fitness, linkage, bal, simplicity, legacy_atomicity, novel_atomicity]:
        metric.set_pRef(pRef)


    two_obj_average_metric = Averager([mean_fitness, linkage])
    three_obj_metric = Averager([simplicity, mean_fitness, linkage])
    normalised_metrics = [two_obj_average_metric, three_obj_metric]



    unnormalised_metrics = [MultipleMetrics([simplicity, mean_fitness, legacy_atomicity]),
                            MultipleMetrics([mean_fitness, novel_atomicity])]

    mutation_operators = [MultimodalMutationOperator(0.5, search_space=pRef.search_space),
                          # SinglePointMutation(chance_of_unfixing=0,
                          #                     probability=1 / pRef.search_space.amount_of_parameters,
                          #                     search_space=pRef.search_space),
                          SinglePointMutation(chance_of_unfixing=0.3,
                                              probability=1 / pRef.search_space.amount_of_parameters,
                                              search_space=pRef.search_space)
                          ]

    selection_operators_normalised = [TruncationSelection(on_aggregated_score=True),
                                      #TournamentSelection()
                                      ]

    selection_operators_unnormalised = [AlternatingSelection([TruncationSelection(which_metric=0),
                                                              TruncationSelection(which_metric=1)])]

    mus = [50] #, 100, 200]
    population_sizes = [1500] #, 500, 1000]
    food_weights = [0.5] # [0.1, 0.5, 0.7]

    mpl = get_variants_for_MPL(pRef=pRef,
                               metrics=normalised_metrics,
                               mutation_operators=mutation_operators,
                               selection_operators=selection_operators_normalised,
                               mus=mus)

    mpl_lexicase = get_variants_for_MPL_lexicase(pRef = pRef,
                                                 metrics=unnormalised_metrics,
                                                 mutation_operators=mutation_operators,
                                                 selection_operators=selection_operators_unnormalised,
                                                 mus=mus)


    mpllr = get_variants_for_MPLLR(pRef = pRef,
                                   metrics=normalised_metrics,
                                   mutation_operators=mutation_operators,
                                   selection_operators=selection_operators_normalised,
                                   mus=mus,
                                   food_weights=food_weights)


    archive = get_variants_for_archive_miner(pRef = pRef,
                                             population_sizes=population_sizes,
                                             metrics = normalised_metrics + unnormalised_metrics)

    return mpl+mpl_lexicase+mpllr+archive


logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
def log(message: str):
    logging.info(message)
def test_variants_on_benchmark_problem(benchmark_problem: BenchmarkProblem,
                                       sample_sizes: list[int],
                                       oversampling_factors: list[int],
                                       evaluation_budgets: list[int]):
    log(f"Beginning to test the variants for {benchmark_problem}, with {sample_sizes = }, {oversampling_factors =}, {evaluation_budgets =}")
    targets = benchmark_problem.get_targets()
    results = []
    for sample_size in sample_sizes:
        for oversampling_factor in oversampling_factors:
            pRef = get_filtered_pRef_from_benchmark_problem(benchmark_problem=benchmark_problem,
                                                            sample_size = sample_size,
                                                            oversampling_factor = oversampling_factor)
            miner_variants = get_all_variants_for_miners(pRef)
            log(f"\tTesting {len(miner_variants)} miner variants on {sample_size = }, {oversampling_factor = }")
            for variant in miner_variants:
                variant.metric.used_evaluations = 0
                accumulated_time = 0
                for evaluation_budget in evaluation_budgets:
                    # note that for different evaluation budgets, the run is simply "restarted" by using the same algorithm instance
                    log(f"\t\tExecuting {variant} with {evaluation_budget = }")
                    try:
                        result_of_run = test_algorithm(variant, evaluation_budget, targets)
                        accumulated_time += result_of_run["runtime"]
                        result_of_run["runtime"] = accumulated_time
                        result_of_run["problem"] = f"{benchmark_problem}"
                        result_of_run["oversampling"] = oversampling_factor
                        results.append(result_of_run)
                    except Exception as e:
                        log(f"Encountered Exception {e}, returning directly")
                        return results

    return results


def run_tests_on_problem(benchmark_problem: BenchmarkProblem,
                         sample_size: int):
    results = test_variants_on_benchmark_problem(benchmark_problem,
                                                 sample_sizes=[sample_size],
                                                 oversampling_factors=[0, 3, 9],
                                                 evaluation_budgets=[15000])
    log("The data has been gathered, results are being dumped")
    print(json.dumps(results, indent=4))



def interpret_problem_str(problem_str):
    match problem_str:
        case "Trap5_5": return Trapk(5, 5)
        case "Trap5_10": return Trapk(10, 5)
        case "RR_5": return RoyalRoad(5, 5)
        case "RR_10": return RoyalRoad(10, 5)
        case "RR0_5": return RoyalRoadWithOverlaps(5, 5, 20)
        case "RR0_10": return RoyalRoadWithOverlaps(10, 5, 40)
        case _: raise Exception(f"Invalid problem string requested: {problem_str}")


def test_problem():
    problem_str = sys.argv[1]
    sample_size = sys.argv[2]
    problem_instance = interpret_problem_str(problem_str)
    run_tests_on_problem(problem_instance, int(sample_size))


def test_all():
    problem_strs = ["Trap5_5", "Trap5_10", "RR_5", "RR_10", "RR0_5", "RR0_10"]
    for problem_str in problem_strs:
        log(f"Testing on {problem_str}")
        problem_instance = interpret_problem_str(problem_str)
        run_tests_on_problem(problem_instance, 100)



