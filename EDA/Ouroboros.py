from math import ceil, sqrt
from typing import Callable, TypeAlias

from BaselineApproaches.Evaluator import Individual
from FullSolution import FullSolution
from PRef import PRef
from PS import PS
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric, MultipleMetrics
from PSMetric.Novelty import Novelty
from PSMiners.ArchiveMiner import ArchiveMiner
from PickAndMerge.PickAndMerge import FSSampler
from SearchSpace import SearchSpace
from TerminationCriteria import EvaluationBudgetLimit


class EXEX(Metric):
    atomicity_evaluator: Metric
    mean_fitness_evaluator: MeanFitness
    novelty_evaluator: Novelty

    def __init__(self):
        super().__init__()

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

    exex_evaluator: EXEX

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
        self.fs_evaluations = 0

        self.mean_fitness_evaluator = MeanFitness()
        self.atomicity_evaluator = Linkage()
        self.novelty_evaluator = Novelty()

        self.current_pRef = PRef.sample_from_search_space(search_space=self.search_space,
                                                          fitness_function=self.fitness_function,
                                                          amount_of_samples=initial_sample_size)
        self.exex_evaluator = EXEX()
        self.exex_evaluator.set_pRef(self.current_pRef)
        self.increment_per_iteration = increment_per_iteration
        self.current_model = []

        self.model_size = model_size

    def calculate_model(self) -> Model:
        budget_limit = EvaluationBudgetLimit(15000)

        miner = ArchiveMiner(150,
                             self.current_pRef,
                             metrics=MultipleMetrics([self.exex_evaluator]))

        miner.run(budget_limit, show_each_generation=True)

        return miner.get_results(quantity_returned=self.model_size)

    def get_extended_pRef(self) -> PRef:
        sampler = FSSampler(self.search_space,
                            individuals=self.current_model,
                            merge_limit=ceil(sqrt(self.search_space.amount_of_parameters)))
        new_samples = [sampler.sample() for _ in range(self.increment_per_iteration)]
        fitnesses = [self.fitness_function(sample) for sample in new_samples]

        newPRef = PRef.from_full_solutions(new_samples, fitnesses, self.search_space)

        return PRef.concat(self.current_pRef, newPRef)



    def step(self):
        self.current_model = self.calculate_model()
        self.current_pRef = self.get_extended_pRef()
        self.exex_evaluator.set_pRef(self.current_pRef)



    def run(self):
        pass # TODO
