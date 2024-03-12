from math import ceil, sqrt
from typing import Callable, TypeAlias

import numpy as np

import utils
from BaselineApproaches.Evaluator import Individual
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from EDA.FSIndividual import FSIndividual
from FullSolution import FullSolution
from PRef import PRef
from PS import PS
from PSMetric.Averager import Averager
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric, MultipleMetrics
from PSMetric.Novelty import Novelty
from PSMetric.SignificantlyHighAverage import SignificantlyHighAverage
from PSMetric.Simplicity import Simplicity
from PSMiners.ArchiveMiner import ArchiveMiner
from PSMiners.FourthMiner import FourthMiner
from PSMiners.MPLSS import MPLSS
from PSMiners.PSMutationOperator import MultimodalMutationOperator
from PickAndMerge.PickAndMerge import FSSampler
from SearchSpace import SearchSpace
from TerminationCriteria import EvaluationBudgetLimit, IterationLimit

Model: TypeAlias = list[Individual]

"""
This class is the EDA which makes use of partial solutions.

Intuitition:
    According to the Schema theorem, a GA works well when the algorithm is implicitly sampling the genes.
    Why not do that explicitly? Those explicit schemata are the partial solutions, which from then on will be called the model.
    
More details:
    The state of the algorithm, between iterations, consists of:
        * the current reference population,
        * The models that were used to construct that reference population:
            * the current exploitation model (beneficial PSs)
            * the current exploration model (novel PS)
    
    
    To transition between states, the algorithm runs as such:
        * Analyse the current reference population to obtain
            * a new exploitative model
            * a new explorative model
        * Sample from those models to obtain a new population
        * Sprinkle in some completely random individuals just to prevent over convergence
    

Current issues:
    * does not archive the purpose of finding good solutions!!!
    * the models are not very clean at all, which defeats the purpose of explainability
    * quite slow
    
"""


class Ouroboros:
    search_space: SearchSpace
    fitness_function: Callable
    fs_evaluations: int
    current_pRef: PRef
    historical_pRef: PRef

    historical_model: Model
    current_exploitative_model: Model
    current_explorative_model: Model

    pRef_sample_size: int
    model_size: int
    elite_size: int

    current_elite: list[FSIndividual]

    def __init__(self,
                 search_space: SearchSpace,
                 fitness_function: Callable,
                 pRef_sample_size: int,
                 model_size: int,
                 elite_size: int):
        self.search_space = search_space
        self.fitness_function = fitness_function
        self.pRef_sample_size = pRef_sample_size
        self.model_size = model_size
        self.elite_size = elite_size
        self.current_elite = []

        self.current_pRef = PRef.sample_from_search_space(search_space=self.search_space,
                                                          fitness_function=self.fitness_function,
                                                          amount_of_samples=pRef_sample_size)
        self.historical_pRef = self.current_pRef
        self.fs_evaluations = self.current_pRef.sample_size

        self.historical_model = []
        self.current_exploitative_model = []
        self.current_explorative_model = []

    def embellish_with_scores(self, individuals: list[Individual], metric: Metric) -> list[Individual]:
        """
        This debug function will add the metric scores onto individuals using the algorithm that generated them.
        This is mainly used to tell which metric is misbehaving when nan values arise
        :param individuals: the individuals which will form a model, freshly generated from the algorithm
        :param algorithm: the algorithm which generated those individuals (or anything with the same metrics!)
        :return: the same individuals, in the same order, but now the .metrics attribute has a list of floats
        """
        assert (isinstance(metric, Averager))
        for individual in individuals:
            individual.metric_scores = metric.get_scores_for_debug(individual.ps)
            return individuals

    @utils.print_entry_and_exit
    def calculate_historical_model(self):
        historical_miner = MPLSS(mu_parameter=20,
                                 lambda_parameter=100,
                                 diversity_offspring_amount=100,
                                 mutation_operator=MultimodalMutationOperator(0.5),
                                 metric=Averager([MeanFitness(), Linkage()]))
        historical_miner.set_pRef(self.historical_pRef)
        historical_miner.run(EvaluationBudgetLimit(15000))

        historical_model = historical_miner.get_results(quantity_returned=self.model_size)
        self.embellish_with_scores(historical_model, historical_miner.metric)
        return historical_model

    @utils.print_entry_and_exit
    def calculate_exploitative_model_classic(self):
        def prepare_metrics():
            mean_fitness = MeanFitness()
            linkage = Linkage()

            mean_fitness.set_pRef(self.current_pRef)
            linkage.set_pRef(self.historical_pRef)  # note how we use the historical pRef here

            return Averager([mean_fitness, linkage])

        exploitation_miner = FourthMiner(population_size=300,
                                         offspring_population_size=300,
                                         metric=prepare_metrics(),
                                         pRef=self.current_pRef)
        exploitation_miner.run(EvaluationBudgetLimit(15000))

        exploitative_model = exploitation_miner.get_results(quantity_returned=self.model_size)
        self.embellish_with_scores(exploitative_model, exploitation_miner.metric)
        return exploitative_model

    @utils.print_entry_and_exit
    def calculate_exploitative_model(self) -> Model:
        """ Generates the PSs which make up the exploitative model """
        exploitation_miner = MPLSS(mu_parameter=20,
                                   lambda_parameter=100,
                                   diversity_offspring_amount=100,
                                   mutation_operator=MultimodalMutationOperator(0.5),
                                   metric=Averager([MeanFitness(), Linkage()]))
        exploitation_miner.set_pRef(self.current_pRef)
        exploitation_miner.run(EvaluationBudgetLimit(15000))

        exploitative_model = exploitation_miner.get_results(quantity_returned=self.model_size)
        self.embellish_with_scores(exploitative_model, exploitation_miner.metric)
        return exploitative_model

    def calculate_exploitative_and_historical_models(self) -> (Model, Model):
        def prepare_metrics():
            mean_fitness = MeanFitness()
            linkage = Linkage()

            mean_fitness.set_pRef(self.current_pRef)
            linkage.set_pRef(self.historical_pRef)  # note how we use the historical pRef here

            return Averager([mean_fitness, linkage])

        metrics = prepare_metrics()
        print("The metrics have been prepared")

        @utils.print_entry_and_exit
        def get_historical_miner():
            historical_miner = MPLSS(mu_parameter=20,
                                     lambda_parameter=100,
                                     diversity_offspring_amount=100,
                                     mutation_operator=MultimodalMutationOperator(0.5),
                                     metric=metrics)
            historical_miner.set_pRef(self.historical_pRef)
            historical_miner.run(EvaluationBudgetLimit(15000))

            historical_model = historical_miner.get_results(quantity_returned=self.model_size)
            self.embellish_with_scores(historical_model, historical_miner.metric)
            return historical_model

        @utils.print_entry_and_exit
        def get_exploitative_miner():
            exploitation_miner = MPLSS(mu_parameter=20,
                                       lambda_parameter=100,
                                       diversity_offspring_amount=100,
                                       mutation_operator=MultimodalMutationOperator(0.5),
                                       metric=metrics)
            exploitation_miner.set_pRef(self.current_pRef)
            exploitation_miner.run(EvaluationBudgetLimit(15000))

            exploitative_model = exploitation_miner.get_results(quantity_returned=self.model_size)
            self.embellish_with_scores(exploitative_model, exploitation_miner.metric)
            return exploitative_model

        history_model = get_historical_miner()
        metrics.used_evaluations = 0  # reset the evaluation counter
        explotation_model = get_exploitative_miner()

        return explotation_model, history_model

    @utils.print_entry_and_exit
    def calculate_explorative_model(self):
        """ Generates the PSs which make up the explorative model """
        exploration_miner = MPLSS(mu_parameter=20,
                                  lambda_parameter=100,
                                  diversity_offspring_amount=20,
                                  mutation_operator=MultimodalMutationOperator(0.5),
                                  metric=Averager([Novelty(), Simplicity()]))
        exploration_miner.set_pRef(self.current_pRef)
        exploration_miner.run(EvaluationBudgetLimit(15000))

        explorative_model = exploration_miner.get_results(quantity_returned=self.model_size)
        self.embellish_with_scores(explorative_model, exploration_miner.metric)
        return explorative_model

    def update_elite(self):
        best_of_current_pref = self.get_best_fs(self.elite_size)

        self.current_elite.extend(best_of_current_pref)
        self.current_elite = list(set(self.current_elite))  # remove duplicates
        self.current_elite.sort(reverse=True)
        self.current_elite = self.current_elite[:self.elite_size]

    def get_pRef_sampled_from_models(self) -> PRef:
        """
        Using the current models, a new reference population is sampled and returned.
        Additionally, some completely random individuals are also added, mainly so that
            * there is no convergence prematurely
            * the linkage information can still be obtained, even when the population is really empty

        Finally, the fss from the current elite are added
        :return: the resulting reference population, disregarding the previous one
        """
        proportion_for_random = 0.2
        amount_from_models = ceil(self.pRef_sample_size * (1 - proportion_for_random))
        amount_from_random = self.pRef_sample_size - amount_from_models

        # add those sampled from the models
        sampler = FSSampler(self.search_space,
                            individuals=self.current_exploitative_model + self.current_explorative_model + self.historical_model,
                            merge_limit=ceil(sqrt(self.search_space.amount_of_parameters)))
        new_samples = [sampler.sample() for _ in range(amount_from_models)]

        # add those sampled randomly
        uniform_random_samples = [FullSolution.random(self.search_space) for _ in range(amount_from_random)]
        new_samples.extend(uniform_random_samples)

        # add those from the elite
        self.update_elite()
        new_samples.extend([fs_individual.full_solution for fs_individual in self.current_elite])

        self.fs_evaluations += len(new_samples)
        fitnesses = [self.fitness_function(sample) for sample in new_samples]

        return PRef.from_full_solutions(new_samples, fitnesses, self.search_space)

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

        def show_elite():
            if len(self.current_elite) == 0:
                print("\tempty")
                return

            for fs_individual in self.current_elite:
                print(f"\t{fs_individual.full_solution}, fitness = {fs_individual.fitness}")

        print(f"The current pRef is \n{state_of_pRef(self.current_pRef)}")
        print(f"The historical pRef is \n{state_of_pRef(self.historical_pRef)}")
        print("The historical model is")
        show_model(self.historical_model)
        print("The exploitative model is")
        show_model(self.current_exploitative_model)
        print("The explorative model is")
        show_model(self.current_explorative_model)
        print("The elite is")
        show_elite()

    def run(self, show_every_generation=False):
        """
        Executes the main iterative loop, which I think is described well enough from the code.
        :param show_every_generation: whether to show debug information regarding every single iteration
        :return: nothing! use .get_best_fs if you want the best full solutions.
        """
        iteration = 0
        iteration_max = 15  # temporary
        while iteration < iteration_max:

            if show_every_generation:
                self.print_current_state()
            self.current_exploitative_model = self.calculate_exploitative_model()

            self.current_exploitative_model, self.historical_model = self.calculate_exploitative_and_historical_models()
            self.current_explorative_model = self.calculate_explorative_model()
            self.historical_model = self.calculate_historical_model()
            self.current_pRef = self.get_pRef_sampled_from_models()
            self.historical_pRef = PRef.concat(self.historical_pRef, self.current_pRef)

    def get_best_fs(self, qty_ret: int) -> list[FSIndividual]:
        indexes_with_scores = list(enumerate(self.current_pRef.fitness_array))
        indexes_with_scores.sort(key=utils.second, reverse=True)

        return [FSIndividual(self.current_pRef.full_solutions[index], score)
                for index, score in indexes_with_scores[:qty_ret]]


def test_ouroboros(benchmark_problem: BenchmarkProblem):
    print("Initialising the EDA")
    eda = Ouroboros(benchmark_problem.search_space,
                    benchmark_problem.fitness_function,
                    pRef_sample_size=10000,
                    model_size=6,
                    elite_size=5)

    eda.run(show_every_generation=True)

    print("The final state is ")
    eda.print_current_state()

    fss = eda.get_best_fs(12)
    print("The Top 12 fss are ")
    for fs_individual in fss:
        fs = fs_individual.full_solution
        as_ps = PS.from_FS(fs)
        fitness = eda.fitness_function(fs)
        print(f"{benchmark_problem.repr_ps(as_ps)}, with fitness = {fitness}")
