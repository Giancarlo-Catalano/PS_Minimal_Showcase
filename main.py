import utils
from BaselineApproaches.FullSolutionGA import FullSolutionGA, test_FSGA
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.IsingSpinGlassProblem import IsingSpinGlassProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.RoyalRoadWithOverlaps import RoyalRoadWithOverlaps
from BenchmarkProblems.SATProblem import SATProblem
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
from PSMiners.MPLSS import test_MLPSS_with_MMM
from PSMiners.MuPlusLambda import test_mu_plus_lambda, test_mu_plus_lambda_with_repeated_trials, \
    test_mu_plus_lambda_with_MMM
from PSMiners.SelfAssembly import test_simple_hill_climber
from PSMiners.ThirdArchiveMiner import test_third_archive_miner
from PickAndMerge.PickAndMerge import test_pick_and_merge
from PyGAD.Testing import test_pygad, test_pygad_on_benchmark_problem
from TerminationCriteria import IterationLimit


def print_separator():
    print("-" * 30)



if __name__ == '__main__':
    # problem = BTProblem.from_files(employee_data_file=r"C:\Users\gac8\PycharmProjects\PS\resources\BT\employeeData.csv",
    #                               rota_file=r"C:\Users\gac8\PycharmProjects\PS\resources\BT\roster_pattern_days.csv",
    #                               calendar_length=56)
    problem = Trapk(6, 5)
    # print("Reading the problem instance")
    # problem = IsingSpinGlassProblem.from_gian_file(
    #    r"C:\Users\gac8\PycharmProjects\PS\resources\IsingSpinGlassInstances\SG_36_1.json")


    # problem = SATProblem.from_json_file(r"C:\Users\gac8\PycharmProjects\PS\resources\SATlibInstances\uf20-01.json")
    # problem = GraphColouring.random(8, 3, 0.3)
    # test_fourth_archive_miner(problem, show_each_generation=True)
    test_ouroboros(problem)
    # test_MLPSS_with_MMM(problem)


    #test_FSGA(problem)
