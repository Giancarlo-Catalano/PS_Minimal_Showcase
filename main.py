#!/usr/bin/env python3

from BaselineApproaches.FullSolutionGA import test_FSGA
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.ToyAmalgam import ToyAmalgam
from GAExplainer.Generational import test_generational
from PSMiners.Archivers.BaselineArchiveMiner import BaselineArchiveMiner
from PSMiners.MuPlusLambda.MPLLR import MPLLR
from PSMiners.MuPlusLambda.MuPlusLambda import MuPlusLambda
from PSMiners.PSMinerDataCollection import run_tests_on_problem, test_all, test_problem


# from MarchMeasurements import test_all


def print_separator():
    print("-" * 30)



if __name__ == '__main__':
    # problem = BTProblem.from_files(employee_data_file=r"C:\Users\gac8\PycharmProjects\PS\resources\BT\employeeData.csv",
    #                               rota_file=r"C:\Users\gac8\PycharmProjects\PS\resources\BT\roster_pattern_days.csv",
    #                               calendar_length=56)
    #problem = ToyAmalgam("KORTP", 5)
    problem = RoyalRoad(5, 5)
    # print("Reading the problem instance")
    # problem = IsingSpinGlassProblem.from_json_file(
    #     r"C:\Users\gac8\PycharmProjects\PS\resources\IsingSpinGlassInstances\SG_36_1.json")
    #problem = MultiDimensionalKnapsack.random(amount_of_dimensions=3, amount_of_items=20, max_value=30)

    # problem = SATProblem.from_json_file(r"C:\Users\gac8\PycharmProjects\PS\resources\SATlibInstances\uf20-01.json")
    # problem = GraphColouring.random(8, 3, 0.3)

    # test_FSGA(problem)
    # test_MLPLR(problem)
    # test_train_tracks_EDA(problem)

    # test_different_atomicities(problem, pRef_size=10000)
    # test_MO_comprehensive(problem)

    #run_tests_on_problem(problem)

    # test_problem()
    # test_generational(problem)

    BaselineArchiveMiner.test_with_problem(problem)
    # MuPlusLambda.test_with_problem(problem)
    MPLLR.test_with_problem(problem)


