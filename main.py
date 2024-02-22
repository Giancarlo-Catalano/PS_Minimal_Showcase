import utils
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.RoyalRoadWithOverlaps import RoyalRoadWithOverlaps
from BenchmarkProblems.Trapk import Trapk
from EDA.Ouroboros import test_ouroboros
from JMetal.PSProblem import test_MO, test_PSProblem, test_MO_comprehensive
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness, ChanceOfGood
from PSMetric.Metric import MultipleMetrics
from PSMetric.Simplicity import Simplicity
from PSMiners.ArchiveMiner import test_archive_miner
from PSMiners.SelfAssembly import test_simple_hill_climber
from PickAndMerge.PickAndMerge import test_pick_and_merge


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
    problem = Trapk(5, 5)
    #test_MO_comprehensive(problem)
    # print_separator()
    # print("Now testing with my own algorithm")

    #metrics = MultipleMetrics([MeanFitness(), Linkage()], weights=[1, 1])
    #test_archive_miner(problem, show_each_generation=True, metrics=metrics, show_interactive_plot=False)

    # test_PSProblem(problem,
    #                which_mo_method="NSGAII",
    #                metrics=metrics,
    #                normalised_objectives=True,
    #                show_interactive_plot=True,
    #                single_objective=False,
    #                save_to_files=True,
    #                max_evaluations=15000)

    test_ouroboros(problem)
