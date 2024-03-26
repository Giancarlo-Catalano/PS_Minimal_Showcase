from PS import PS
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
    final_population = [individual.ps for individual in final_population]
    indexes_of_targets = [final_population.index(target) for target in target_PS]

    results_of_run["indexes_of_targets"] = indexes_of_targets

    results_of_run["parameters"] = algorithm_instance.get_parameters_as_dict()

    return results_of_run
