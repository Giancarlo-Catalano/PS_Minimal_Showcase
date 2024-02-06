from typing import Callable

import TerminationCriteria
from BaselineApproaches.Evaluator import PSEvaluator
from BaselineApproaches.FullSolutionGA import FullSolutionGA
from BaselineApproaches.PSGA import PSGA
from FullSolution import FullSolution
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from PSMiners.ArchiveBasedSpecialisationMiner import ABSM
from SearchSpace import SearchSpace
from custom_types import Fitness


def test_fsga(search_space: SearchSpace, fitness_function: Callable):
    budget_limit = TerminationCriteria.EvaluationBudgetLimit(10000)
    found_global = TerminationCriteria.UntilGlobalOptimaReached(3)
    termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, found_global)

    fs_ga = FullSolutionGA(search_space=search_space,
                           crossover_rate=0.5,
                           mutation_rate=0.1,
                           elite_size=2,
                           tournament_size=3,
                           population_size=100,
                           fitness_function=fitness_function)

    fs_ga.run(termination_criteria)

    results = fs_ga.get_best_of_population(amount=3)
    print("The results are:")
    for fs, fitness in results:
        print(f"FS: {fs}, fitness = {fitness}")


if __name__ == '__main__':
    search_space = SearchSpace([2, 2, 2, 2, 2, 2])


    def fitness_function(fs: FullSolution) -> Fitness:
        return float(fs.values[0] * fs.values[1] * fs.values[2] + (1 - fs.values[3]) * (1 - fs.values[4]) + fs.values[5])


    pRef: PRef = PRef.sample_from_search_space(search_space, fitness_function, 1000)

    # print(f"The pRef is {pRef.long_repr()}")

    simplicity = Simplicity()
    mean_fitness = MeanFitness()
    atomicity = Atomicity()

    ps_evaluator = PSEvaluator([simplicity, mean_fitness, atomicity], pRef)

    budget_limit = TerminationCriteria.EvaluationBudgetLimit(10000)
    iteration_limit = TerminationCriteria.IterationLimit(12)
    termination_criteria = TerminationCriteria.UnionOfCriteria(budget_limit, iteration_limit)

    ps_ga = PSGA(search_space=search_space,
                 crossover_rate=0.5,
                 mutation_rate=0.1,
                 elite_size=2,
                 tournament_size=3,
                 population_size=100,
                 ps_evaluator=ps_evaluator)

    absm = ABSM(150, ps_evaluator)

    absm.run(termination_criteria)

    results = absm.get_best_of_last_run(quantity_returned=10)
    print("The results are:")
    for ps, fitness in results:
        print(f"PS: {ps}, fitness = {fitness}")
