import utils
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.RoyalRoadWithOverlaps import RoyalRoadWithOverlaps
from BenchmarkProblems.Trapk import Trapk
from JMetal.PSProblem import test_MO, test_PSProblem, test_MO_comprehensive
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from PSMiners.ArchiveMiner import test_archive_miner
from PSMiners.SelfAssembly import test_simple_hill_climber


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
    problem = Trapk(5, 5)
    #problem = RoyalRoadWithOverlaps(4, 4, 15)
    #test_MO_comprehensive(problem)
    #print_separator()
    #print("Now testing with my own algorithm")
    test_archive_miner(problem, efficient=True, show_each_generation=True)
