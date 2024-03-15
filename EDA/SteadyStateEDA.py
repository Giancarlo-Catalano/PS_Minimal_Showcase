from math import ceil, sqrt
from typing import TypeAlias, Callable, Any

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
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric
from PSMetric.NoveltyFromPopulation import NoveltyFromPopulation
from PSMetric.Simplicity import Simplicity
from PSMiners.MPLSS import MPLSS
from PSMiners.PSMutationOperator import MultimodalMutationOperator
from PickAndMerge.PickAndMerge import FSSampler
from SearchSpace import SearchSpace
from TerminationCriteria import EvaluationBudgetLimit, TerminationCriteria, IterationLimit

Model: TypeAlias = list[Individual]
Fitness: TypeAlias = float
FitnessFunction: TypeAlias = Callable[[FullSolution], Fitness]



def merge_populations(first: list[Any], second: list[Any], quantity_returned: int,
                      remove_duplicates=False) -> list[Any]:
    """Given a population of orderable items, it merges them and returns the top n """
    returned_population = first + second
    if remove_duplicates:
        returned_population = list(set(returned_population))
    returned_population.sort(reverse = True)
    return returned_population[:quantity_returned]



class SteadyStateEDA:
    current_pRef: PRef
    historical_pRef: PRef

    cutting_edge_model: Model
    novelty_model: Model
    historical_model: Model

    linkage_metric: BiVariateANOVALinkage
    mean_fitness_metric: MeanFitness
    novelty_metric: NoveltyFromPopulation
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
        self.current_pRef = self.fitness_function_evaluator.generate_pRef_from_search_space(self.search_space, self.population_size)
        self.historical_pRef = self.current_pRef

        self.cutting_edge_model = []
        self.novelty_model = []
        self.historical_model = []

        self.linkage_metric = BiVariateANOVALinkage()
        self.mean_fitness_metric = MeanFitness()

        self.novelty_metric = NoveltyFromPopulation()
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

    def update_metrics_and_linkage_model(self):
        self.linkage_metric.set_pRef(self.current_pRef)  # maybe put historical here
        self.mean_fitness_metric.set_pRef(self.current_pRef) # and here
        self.novelty_metric.set_pRef(self.current_pRef)
        self.simplicity_metric.set_pRef(self.current_pRef)


    def get_cutting_edge_model(self):
        ce_miner = MPLSS(mu_parameter=20,
                                   lambda_parameter=100,
                                   diversity_offspring_amount=100,
                                   mutation_operator=MultimodalMutationOperator(0.5),
                                   metric=Averager([self.mean_fitness_metric, self.linkage_metric]))
        ce_miner.set_pRef(self.current_pRef, set_metrics=False)
        ce_miner.run(EvaluationBudgetLimit(15000))

        ce_model = ce_miner.get_results(quantity_returned=self.model_size)
        return ce_model

    def get_novelty_model(self):
        novelty_miner = MPLSS(mu_parameter=20,
                                   lambda_parameter=100,
                                   diversity_offspring_amount=100,
                                   mutation_operator=MultimodalMutationOperator(0.5),
                                   metric=Averager([self.novelty_metric, self.simplicity_metric]))
        novelty_miner.set_pRef(self.current_pRef, set_metrics=False)
        novelty_miner.run(EvaluationBudgetLimit(15000))

        novelty_model = novelty_miner.get_results(quantity_returned=self.model_size)
        return novelty_model


    def update_historical_model(self):
        # TODO: the scores in the historical model are now invalid, because the population has changed.
        # perhaps we need to keep a history_pRef?
        historical_metric = Averager([self.mean_fitness_metric, self.linkage_metric])
        for individual in self.historical_model:
            individual.aggregated_score = historical_metric.get_single_normalised_score(individual.ps)

        self.historical_model = merge_populations(self.historical_model, self.cutting_edge_model,
                                                  quantity_returned=self.model_size)

    def sample_from_models(self) -> PRef:
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

        return self.fitness_function_evaluator.generate_pRef_from_full_solutions(self.search_space, new_samples)
    def update_models_using_current_population(self):
        self.update_metrics_and_linkage_model()
        self.cutting_edge_model = self.get_cutting_edge_model()
        self.novelty_model = self.get_novelty_model()
        self.update_historical_model()

    def update_population_using_current_models(self):
        self.current_pRef = self.sample_from_models()
        self.historical_pRef = PRef.concat(self.historical_pRef, self.current_pRef)


    def run(self,
            termination_criteria: TerminationCriteria,
            show_every_iteration = False):
        iteration = 0
        def should_terminate():
            return termination_criteria.met(iterations = iteration,
                                            used_budget = self.fitness_function_evaluator.used_evaluations)

        while not should_terminate():
            if show_every_iteration:
                self.print_current_state()

            self.update_models_using_current_population()
            self.update_population_using_current_models()


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
                         population_size=10000,
                         offspring_size=1000,
                         model_size=12)

    eda.run(show_every_iteration=True,
            termination_criteria=IterationLimit(12))

    print("The final state is ")
    eda.print_current_state()

    fss = eda.get_results[:12]
    print("The Top 12 fss are ")
    for fs_individual in fss:
        fs = fs_individual.full_solution
        as_ps = PS.from_FS(fs)
        fitness = eda.fitness_function(fs)
        print(f"{benchmark_problem.repr_ps(as_ps)}, with fitness = {fitness}")













