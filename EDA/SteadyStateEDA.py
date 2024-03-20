from math import ceil, sqrt
from typing import TypeAlias, Callable, Any

import numpy as np

import utils
from BaselineApproaches.Evaluator import Individual
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from EDA.FSEvaluator import FSEvaluator
from EDA.FSIndividual import FSIndividual
from FullSolution import FullSolution
from PRef import PRef
from PS import PS
from PSMetric.Averager import Averager
from PSMetric.BiVariateANOVALinkage import BiVariateANOVALinkage
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric
from PSMetric.NoveltyFromModel import NoveltyFromModel
from PSMetric.NoveltyFromPopulation import NoveltyFromPopulation
from PSMetric.Simplicity import Simplicity
from PSMiners.MPLLR import MPLLR
from PSMiners.MPLSS import MPLSS
from PSMiners.PSMutationOperator import MultimodalMutationOperator
from PickAndMerge.PickAndMerge import FSSampler
from SearchSpace import SearchSpace
from TerminationCriteria import EvaluationBudgetLimit, TerminationCriteria, IterationLimit, PSEvaluationLimit

Model: TypeAlias = list[Individual]
Fitness: TypeAlias = float
FitnessFunction: TypeAlias = Callable[[FullSolution], Fitness]


def merge_populations(first: list[Any], second: list[Any], quantity_returned: int,
                      remove_duplicates=False) -> list[Any]:
    """Given a population of orderable items, it merges them and returns the top n """
    returned_population = first + second
    if remove_duplicates:
        returned_population = list(set(returned_population))
    returned_population.sort(reverse=True)
    return returned_population[:quantity_returned]


class SteadyStateEDA:
    current_pRef: PRef
    # historical_pRef: PRef

    cutting_edge_model: Model
    novelty_model: Model
    historical_model: Model

    linkage_metric: BiVariateANOVALinkage
    mean_fitness_metric: MeanFitness
    novelty_metric: NoveltyFromModel
    simplicity_metric: Simplicity

    # main parameters for EDA
    population_size: int
    offspring_size: int
    model_size: int

    # problem parameters
    search_space: SearchSpace
    fitness_function_evaluator: FSEvaluator

    def __init__(self,
                 search_space: SearchSpace,
                 fitness_function: FitnessFunction,
                 population_size: int,
                 offspring_size: int,
                 model_size: int):
        self.search_space = search_space
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.model_size = model_size

        self.fitness_function_evaluator = FSEvaluator(fitness_function)
        self.current_pRef = self.fitness_function_evaluator.generate_pRef_from_search_space(self.search_space,
                                                                                            self.population_size)
        # self.historical_pRef = self.current_pRef

        self.cutting_edge_model = []
        self.novelty_model = []
        self.historical_model = []

        self.linkage_metric = BiVariateANOVALinkage()
        self.mean_fitness_metric = MeanFitness()

        self.novelty_metric = NoveltyFromModel()
        self.simplicity_metric = Simplicity()

    def print_current_state(self):
        def state_of_pRef(pRef):
            stats = utils.get_descriptive_stats(pRef.fitness_array)
            return (f"\t#samples = {pRef.sample_size}, "
                    f"\n\tmin={stats[0]:.2f}, median={stats[1]:.2f}, max={stats[2]:.2f}, "
                    f"\n\tavg={stats[3]:.2f}, stdev={stats[4]:.2f}")

        def show_model(model: Model):
            if len(model) == 0:
                print("\tempty")
            else:
                for item in model:
                    print(f"\t{item.ps}, score = {item.aggregated_score:.3f}")

        print(f"The current pRef is \n{state_of_pRef(self.current_pRef)}")
        print("The historical model is")
        show_model(self.historical_model)
        print("The cutting edge model is")
        show_model(self.cutting_edge_model)
        print("The novelty model is")
        show_model(self.novelty_model)


    def get_current_state_as_dict(self):
        def get_pRef_dict(pRef):
            stats = utils.get_descriptive_stats(pRef.fitness_array)
            return {"min": stats[0],
                    "median": stats[1],
                    "max": stats[2],
                    "avg": stats[3],
                    "stdev": stats[4]}

        ps_evals = self.mean_fitness_metric.used_evaluations
        fs_evals = self.fitness_function_evaluator.used_evaluations
        return {"pRef": get_pRef_dict(self.current_pRef),
                "ps_evals": ps_evals,
                "fs_evals": fs_evals}

    def reset_ps_metrics_evaluation_counters(self):
        self.mean_fitness_metric.used_evaluations = 0
        self.novelty_metric.used_evaluations = 0
        # the other 2 don't matter

    def update_metrics_and_linkage_model(self):
        self.linkage_metric.set_pRef(self.current_pRef)   # maybe history_pRef would be better here
        self.mean_fitness_metric.set_pRef(self.current_pRef)
        self.novelty_metric.set_pRef(self.current_pRef)
        self.simplicity_metric.set_pRef(self.current_pRef)

    def get_cutting_edge_model(self, termination_criteria: TerminationCriteria):
        self.reset_ps_metrics_evaluation_counters()
        # print("Generating Exploitative model...",end="")
        ce_miner = MPLLR(mu_parameter=50,
                         lambda_parameter=300,
                         food_weight=0.3,
                         mutation_operator=MultimodalMutationOperator(0.5),
                         metric=Averager([self.mean_fitness_metric, self.linkage_metric]),
                         starting_population = self.novelty_model)
        ce_miner.set_pRef(self.current_pRef, set_metrics=False)
        ce_miner.run(termination_criteria)

        ce_model = ce_miner.get_results(quantity_returned=self.model_size)
        # print("Finished!")
        return ce_model

    def get_novelty_model(self, termination_criteria: TerminationCriteria):
        self.reset_ps_metrics_evaluation_counters()
        # print("Generating Novelty model...", end="")
        self.novelty_metric.set_reference_model(self.historical_model + self.cutting_edge_model)
        novelty_miner = MPLLR(mu_parameter=20,
                              lambda_parameter=100,
                              food_weight=0.3,
                              mutation_operator=MultimodalMutationOperator(0.5),
                              metric=Averager([self.novelty_metric, self.simplicity_metric]))
        novelty_miner.set_pRef(self.current_pRef, set_metrics=False)
        novelty_miner.run(termination_criteria)

        novelty_model = novelty_miner.get_results(quantity_returned=self.model_size)
        # print("Finished!")
        return novelty_model

    def update_historical_model(self):
        historical_metric = Averager([self.mean_fitness_metric, self.linkage_metric])
        for individual in self.historical_model:
            individual.aggregated_score = historical_metric.get_single_normalised_score(individual.ps)

        self.historical_model = merge_populations(self.historical_model, self.cutting_edge_model,
                                                  quantity_returned=self.model_size)

    def generate_new_solutions(self) -> PRef:
        proportion_for_random = 0.2
        amount_from_models = ceil(self.offspring_size * (1 - proportion_for_random))
        amount_from_random = self.offspring_size - amount_from_models

        # add those sampled from the models
        sampler = FSSampler(self.search_space,
                            individuals=self.cutting_edge_model + self.novelty_model + self.historical_model,
                            merge_limit=ceil(sqrt(self.search_space.amount_of_parameters)))
        new_samples = [sampler.sample() for _ in range(amount_from_models)]

        # add those sampled randomly
        uniform_random_samples = [FullSolution.random(self.search_space) for _ in range(amount_from_random)]
        new_samples.extend(uniform_random_samples)

        return self.fitness_function_evaluator.generate_pRef_from_full_solutions(
            search_space=self.current_pRef.search_space,
            samples=new_samples)

    def update_models_using_current_population(self, ps_termination_criteria: TerminationCriteria):
        self.update_metrics_and_linkage_model()
        self.cutting_edge_model = self.get_cutting_edge_model(ps_termination_criteria)
        self.novelty_model = self.get_novelty_model(ps_termination_criteria)

    def merge_pRefs(self, original_pRef: PRef, new_pRef: PRef) -> PRef:
        # very inefficient at the moment
        amount_to_keep = self.population_size
        all_fitnesses = list(enumerate(np.concatenate((original_pRef.fitness_array, new_pRef.fitness_array))))
        all_fitnesses.sort(key=utils.second, reverse=True)

        kept_indexes, kept_fitnesses = utils.unzip(all_fitnesses[:amount_to_keep])
        all_solutions = original_pRef.full_solutions + new_pRef.full_solutions

        kept_solutions = [all_solutions[i] for i in kept_indexes]
        return PRef.from_full_solutions(kept_solutions,
                                        fitness_values=kept_fitnesses,
                                        search_space=original_pRef.search_space)

    def update_population_using_current_models(self):
        new_pRef = self.generate_new_solutions()
        self.current_pRef = self.merge_pRefs(self.current_pRef, new_pRef)
        # self.historical_pRef = PRef.concat(self.historical_pRef, new_pRef)

    def run(self,
            fs_termination_criteria: TerminationCriteria,
            ps_termination_criteria: TerminationCriteria,
            show_every_iteration=False):

        logger_dict = {"iterations": []}
        iteration = 0

        def should_terminate():
            return fs_termination_criteria.met(iterations=iteration,
                                               fs_evaluations=self.fitness_function_evaluator.used_evaluations)

        while not should_terminate():
            if show_every_iteration:
                self.print_current_state()

            logger_dict["iterations"].append({"iteration": iteration,
                                              "state": self.get_current_state_as_dict()})

            self.update_models_using_current_population(ps_termination_criteria)
            self.update_population_using_current_models()
            self.update_historical_model()

            iteration +=1
        logger_dict["iterations"].append({"iteration": iteration,
                                          "state": self.get_current_state_as_dict()})
        return logger_dict

    def get_results(self):
        population = [FSIndividual(fs, fitness)
                      for fs, fitness in zip(self.current_pRef.full_solutions,
                                             self.current_pRef.fitness_array)]
        population.sort(reverse=True)
        return population


def test_sseda(benchmark_problem: BenchmarkProblem):
    print("Initialising the SS EDA")
    eda = SteadyStateEDA(search_space=benchmark_problem.search_space,
                         fitness_function=benchmark_problem.fitness_function,
                         population_size=1000,
                         offspring_size=1000,
                         model_size=12)

    eda.run(show_every_iteration=True,
            fs_termination_criteria=IterationLimit(12),
            ps_termination_criteria=PSEvaluationLimit(10000))

    print("The final state is ")
    eda.print_current_state()

    fss = eda.get_results[:12]
    print("The Top 12 fss are ")
    for fs_individual in fss:
        fs = fs_individual.full_solution
        as_ps = PS.from_FS(fs)
        fitness = eda.fitness_function(fs)
        print(f"{benchmark_problem.repr_ps(as_ps)}, with fitness = {fitness}")
