from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from JMetal.PSProblem import test_MO
from PSMiners.ArchiveMiner import test_archive_miner
from PSMiners.SelfAssembly import test_simple_hill_climber


def print_separator():
    print("-" * 30)


def test_many_miners():
    benchmark_problem = RoyalRoad(4, 5)
    # print(f"The problem is {problem.long_repr()}")
    print_separator()
    test_archive_miner(benchmark_problem)
    print_separator()
    test_MO(benchmark_problem)
    print_separator()
    test_simple_hill_climber(benchmark_problem)


if __name__ == '__main__':
    problem = Trapk(3, 5)
    test_many_miners()
