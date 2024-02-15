import utils
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from JMetal.PSProblem import test_MO, test_PSProblem, test_MO_comprehensive
from PSMetric.KindaAtomicity import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from PSMiners.ArchiveMiner import test_archive_miner
from PSMiners.SelfAssembly import test_simple_hill_climber


def print_separator():
    print("-" * 30)


def test_many_miners():
    benchmark_problem = RoyalRoad(4, 5)
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
    algorithms = ["NSGAII", "MOCell", "GDE3"]
    problem = Trapk(3, 5)
    print("Testing with a single objective")
    for algorithm in algorithms:
        print(f"\n\nTesting with {algorithm}")
        test_PSProblem(problem,
                       which_mo_method=algorithm,
                       single_objective=True,
                       save_to_files=False)
