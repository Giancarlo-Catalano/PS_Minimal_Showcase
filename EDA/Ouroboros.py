from math import ceil, sqrt
from typing import Callable, TypeAlias

import utils
from BaselineApproaches.Evaluator import Individual
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FullSolution import FullSolution
from PRef import PRef
from PS import PS
from PSMetric.Averager import Averager
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric, MultipleMetrics
from PSMetric.Novelty import Novelty
from PSMiners.ArchiveMiner import ArchiveMiner
from PSMiners.FourthMiner import FourthMiner
from PickAndMerge.PickAndMerge import FSSampler
from SearchSpace import SearchSpace
from TerminationCriteria import EvaluationBudgetLimit, IterationLimit


class EXEX(Metric):
    atomicity_evaluator: Metric
    mean_fitness_evaluator: MeanFitness
    novelty_evaluator: Novelty

    def __init__(self):
        super().__init__()
        self.atomicity_evaluator = Linkage()
        self.mean_fitness_evaluator = MeanFitness()
        self.novelty_evaluator = Novelty()

    def __repr__(self):
        return "Exploration||Exploitation"

    def set_pRef(self, pRef: PRef):
        self.atomicity_evaluator.set_pRef(pRef)
        self.mean_fitness_evaluator.set_pRef(pRef)
        self.novelty_evaluator.set_pRef(pRef)

    def get_single_normalised_score(self, ps: PS) -> float:
        mean_fitness = self.mean_fitness_evaluator.get_single_normalised_score(ps)
        atomicity = self.atomicity_evaluator.get_single_normalised_score(ps)
        novelty = self.novelty_evaluator.get_single_normalised_score(ps)

        exploitation = (mean_fitness + atomicity) / 2.0
        return max(exploitation, novelty)

Model: TypeAlias = list[Individual]

class Ouroboros:
    search_space: SearchSpace
    fitness_function: Callable
    fs_evaluations: int
    current_pRef: PRef
    current_model: list[Individual]

    exploitative_evaluator: Averager
    explorative_evaluator: Novelty

    increment_per_iteration: int
    model_size: int

    def __init__(self,
                 search_space: SearchSpace,
                 fitness_function: Callable,
                 initial_sample_size: int,
                 increment_per_iteration: int,
                 model_size: int):
        self.search_space = search_space
        self.fitness_function = fitness_function

        self.current_pRef = PRef.sample_from_search_space(search_space=self.search_space,
                                                          fitness_function=self.fitness_function,
                                                          amount_of_samples=initial_sample_size)
        self.fs_evaluations = self.current_pRef.sample_size
        self.exploitative_evaluator = Averager([MeanFitness(), Linkage()])
        self.exploitative_evaluator.set_pRef(self.current_pRef)
        self.explorative_evaluator = Novelty()
        self.explorative_evaluator.set_pRef(self.current_pRef)
        self.increment_per_iteration = increment_per_iteration
        self.current_model = []

        self.model_size = model_size

    def calculate_model(self) -> Model:
        termination_criteria = IterationLimit(12)

        exploitation_miner = FourthMiner(population_size=150,
                                         pRef=self.current_pRef,
                                         offspring_population_size=450,
                                         metric = Averager([MeanFitness(), Linkage()]))

        exploitation_miner.run(termination_criteria)

        exploration_miner = FourthMiner(population_size=150,
                                        pRef=self.current_pRef,
                                        offspring_population_size=450,
                                        metric = self.explorative_evaluator)

        exploration_miner.run(termination_criteria)

        exploitative_model = exploitation_miner.get_results(quantity_returned=self.model_size)
        explorative_model = exploration_miner.get_results(quantity_returned=self.model_size)
        return exploitative_model + explorative_model

    def get_extended_pRef(self) -> PRef:
        sampler = FSSampler(self.search_space,
                            individuals=self.current_model,
                            merge_limit=ceil(sqrt(self.search_space.amount_of_parameters)))
        new_samples = [sampler.sample() for _ in range(self.increment_per_iteration)]
        self.fs_evaluations += len(new_samples)
        fitnesses = [self.fitness_function(sample) for sample in new_samples]

        newPRef = PRef.from_full_solutions(new_samples, fitnesses, self.search_space)

        return PRef.concat(self.current_pRef, newPRef)

    def step(self):
        self.current_model = self.calculate_model()
        self.current_pRef = self.get_extended_pRef()
        self.exploitative_evaluator.set_pRef(self.current_pRef)
        self.explorative_evaluator.set_pRef(self.current_pRef)

    def print_current_state(self):
        print(f"The pRef has {self.current_pRef.sample_size}, and the model is ")
        for item in self.current_model:
            mean_fitness, atomicity_fitness = self.exploitative_evaluator.get_normalised_scores(item.ps)
            novelty = self.explorative_evaluator.get_single_normalised_score(item.ps)

            print(f"\t{item}, mf = {mean_fitness:.3f}, atomicity = {atomicity_fitness:.3f}, novelty = {novelty:.3f}")

    def run(self, show_every_generation=False):
        iteration = 0
        iteration_max = 12
        while iteration < iteration_max:

            if show_every_generation:
                self.print_current_state()
            self.step()

    def get_best_fs(self, qty_ret: int):
        indexes_with_scores = list(enumerate(self.current_pRef.fitness_array))
        indexes_with_scores.sort(key=utils.second, reverse=True)

        return [self.current_pRef.full_solutions[index] for index, score in indexes_with_scores[:qty_ret]]


def test_ouroboros(benchmark_problem: BenchmarkProblem):
    print("Initialising the EDA")
    eda = Ouroboros(benchmark_problem.search_space,
                    benchmark_problem.fitness_function,
                    initial_sample_size=1000,
                    increment_per_iteration=1000,
                    model_size=20)

    eda.run(show_every_generation=True)

    print("The final model is ")
    for individual in eda.current_model:
        print(f"{individual}")

    fss = eda.get_best_fs(12)
    print("The Top 12 fss are ")
    for fs in fss:
        print(f"{fs}, with fitness = {eda.fitness_function(fs)}")
