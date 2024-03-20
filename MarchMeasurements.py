import json

import TerminationCriteria
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.IsingSpinGlassProblem import IsingSpinGlassProblem
from BenchmarkProblems.OneMax import OneMax
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.SATProblem import SATProblem
from BenchmarkProblems.Trapk import Trapk
from EDA.SteadyStateEDA import SteadyStateEDA
from utils import execution_time

is_on_cluster = False

if is_on_cluster:
    resources_root = r"/home/gac/EDA/PS/resources/"
    ising_root = resources_root + "IsingSpinGlassInstances/"
    satlib_root = resources_root + "3sat/"
else:
    resources_root = r"C:\Users\gac8\PycharmProjects\PS\resources" + "\\"
    ising_root = resources_root + "IsingSpinGlassInstances" + "\\"
    satlib_root = resources_root + "3sat" + "\\"

problems = {"Trap5_3": Trapk(3, 5),
            "Trap5_5": Trapk(5, 5),
            "Trap5_10": Trapk(10, 5),
            "Trap5_20": Trapk(20, 5),
            "RoyalRoad5_3": RoyalRoad(3, 5),
            "RoyalRoad5_5": RoyalRoad(5, 5),
            "RoyalRoad5_10": RoyalRoad(10, 5),
            "RoyalRoad5_20": RoyalRoad(20, 5),
            "Ising4": IsingSpinGlassProblem.from_gian_file(ising_root + "SG_16_1.json"),
            "Ising8": IsingSpinGlassProblem.from_gian_file(ising_root + "SG_64_1.json"),
            "Ising16": IsingSpinGlassProblem.from_gian_file(ising_root + "SG_256_1.json"),
            "Ising20": IsingSpinGlassProblem.from_gian_file(ising_root + "SG_400_1.json"),
            "OneMax10": OneMax(10),
            "OneMax50": OneMax(50),
            "OneMax100": OneMax(100),
            "SAT20": SATProblem.from_json_file(satlib_root + "uf20-01.json")
            }

population_sizes = [50, 100, 200, 500]
fs_budgets = [10000] # , 15000, 20000, 30000]
ps_budgets = [200] # [5000, 10000, 15000]
model_sizes = [6] # [6, 12, 24]


def test_EDA_with_parameters(problem: BenchmarkProblem,
                             problem_string: str,
                             population_size: int,
                             fs_budget: int,
                             ps_budget: int,
                             model_size: int):
    algorithm = SteadyStateEDA(search_space=problem.search_space,
                               fitness_function=problem.fitness_function,
                               population_size=population_size,
                               offspring_size=population_size,
                               model_size=model_size)

    fs_evaluation_limit = TerminationCriteria.FullSolutionEvaluationLimit(fs_budget)
    ps_evaluation_limit = TerminationCriteria.PSEvaluationLimit(ps_budget)

    log_dict = dict()
    log_dict["problem"] = problem_string
    log_dict["population_size"] = population_size
    log_dict["fs_budget"] = fs_budget
    log_dict["ps_budget"] = ps_budget
    log_dict["model_size"] = model_size

    with execution_time() as runtime:
        run_data = algorithm.run(fs_termination_criteria=fs_evaluation_limit,
                                 ps_termination_criteria=ps_evaluation_limit)

    log_dict["runtime"] = runtime.execution_time
    log_dict["run_data"] = run_data

    final_best_fitness = algorithm.get_results()[0].fitness
    log_dict["best_fitness"] = float(final_best_fitness)
    log_dict["reached_global_optima"] = bool(final_best_fitness == problem.get_global_optima_fitness())  # bool_ is not serializable


    return log_dict



def test_all():
    results = [test_EDA_with_parameters(problem, problem_string, population_size, fs_budget, ps_budget, model_size)
               for problem_string, problem in list(problems.items())[:1]
               for population_size in population_sizes
               for fs_budget in fs_budgets
               for ps_budget in ps_budgets
               for model_size in model_sizes]

    print(json.dumps(results, indent=4))
