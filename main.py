import utils
from BaselineApproaches.FullSolutionGA import FullSolutionGA
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.IsingSpinGlassProblem import IsingSpinGlassProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.RoyalRoadWithOverlaps import RoyalRoadWithOverlaps
from BenchmarkProblems.Trapk import Trapk
from EDA.Ouroboros import test_ouroboros
from JMetal.PSProblem import test_MO, test_PSProblem, test_MO_comprehensive
from JMetal.SingleObjective import test_single_objective_search
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness, ChanceOfGood
from PSMetric.Metric import MultipleMetrics
from PSMetric.Simplicity import Simplicity
from PSMiners.ArchiveMiner import test_archive_miner
from PSMiners.EfficientArchiveMiner import test_efficient_archive_miner
from PSMiners.FourthMiner import test_fourth_archive_miner
from PSMiners.SelfAssembly import test_simple_hill_climber
from PSMiners.ThirdArchiveMiner import test_third_archive_miner
from PickAndMerge.PickAndMerge import test_pick_and_merge
from PyGAD.Testing import test_pygad, test_pygad_on_benchmark_problem
from TerminationCriteria import IterationLimit


def print_separator():
    print("-" * 30)


def test_many_miners():
    benchmark_problem = RoyalRoadWithOverlaps(4, 5, 15)
    # print(f"The problem is {problem.long_repr()}")
    print_separator()
    test_archive_miner(benchmark_problem, False)
    print_separator()
    test_archive_miner(benchmark_problem, True)
    print_separator()
    test_MO(benchmark_problem)
    print_separator()
    test_simple_hill_climber(benchmark_problem)


if __name__ == '__main__':
    # problem = BTProblem.from_files(employee_data_file=r"C:\Users\gac8\PycharmProjects\PS\resources\BT\employeeData.csv",
    #                               rota_file=r"C:\Users\gac8\PycharmProjects\PS\resources\BT\roster_pattern_days.csv",
    #                               calendar_length=56)
    # problem = Trapk(5, 5)
    print("Reading the problem instance")
    problem = IsingSpinGlassProblem.from_gian_file(
        r"C:\Users\gac8\PycharmProjects\PS\resources\IsingSpinGlassInstances\SG_25_1.json")
    # test_MO_comprehensive(problem)
    # print_separator()
    # print("Now testing with my own algorithm")

    # metrics = MultipleMetrics([MeanFitness(), Linkage()], weights=[1, 1])
    # test_fourth_archive_miner(problem, show_each_generation=True)
    test_ouroboros(problem)

    # test_single_objective_search(problem, evaluation_budget=15000)

    # test_pygad_on_benchmark_problem(problem)
    # test_PSProblem(benchmark_problem = problem,
    #                which_mo_method = "NSGAII",
    #                metrics=MultipleMetrics([MeanFitness(), Linkage()]),
    #                normalised_objectives=True,
    #                show_interactive_plot=False,
    #                single_objective=False,
    #                save_to_files=True,
    #                evaluation_budget=15000)

    # print("Reading the problem instance")
    # problem = IsingSpinGlassProblem.from_gian_file(
    #     r"C:\Users\gac8\PycharmProjects\PS\resources\IsingSpinGlassInstances\SG_16_1.json")
    #
    #
    #
    # print("Initialing the GA")
    # fs_ga = FullSolutionGA(search_space=problem.search_space,
    #                        mutation_rate=1 / problem.amount_of_variables,
    #                        crossover_rate=0.5,
    #                        elite_size=2,
    #                        tournament_size=3,
    #                        population_size=100,
    #                        fitness_function=problem.fitness_function)
    #
    # termination_criteria = IterationLimit(30)
    #
    # print("Running the GA")
    # fs_ga.run(termination_criteria)
    #
    # results = fs_ga.get_results()
    #
    # for fs, score in results:
    #     print(f"{fs} with score {score}")


