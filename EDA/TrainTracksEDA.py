import heapq
from math import ceil, sqrt
from typing import Callable, TypeAlias

import numpy as np

import TerminationCriteria
import utils
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
from PSMiners.Individual import Individual
from PSMiners.MPLLR import MPLLR
from PSMiners.PSMutationOperator import PSMutationOperator, MultimodalMutationOperator
from PickAndMerge.PickAndMerge import FSSampler
from SearchSpace import SearchSpace

PSIndividual: TypeAlias = Individual


class TrainTracksEDA:
    current_population: list[FSIndividual]
    current_model: list[PSIndividual]

    search_space: SearchSpace
    fitness_function_evaluator: FSEvaluator

    fs_population_size: int
    fs_offspring_size: int

    ps_model_size: int
    ps_model_selected_amount: int
    ps_mutation_operator: PSMutationOperator

    def __init__(self,
                 search_space: SearchSpace,
                 fitness_function: Callable,
                 fs_population_size: int,
                 fs_offspring_size: int,
                 ps_model_size: int,
                 ps_model_mu: int,
                 ps_mutation_operator: PSMutationOperator):
        self.search_space = search_space
        self.fitness_function_evaluator = FSEvaluator(fitness_function)
        self.fs_population_size = fs_population_size
        self.fs_offspring_size = fs_offspring_size
        self.ps_model_size = ps_model_size
        self.ps_model_selected_amount = ps_model_mu
        self.ps_mutation_operator = ps_mutation_operator
        self.ps_mutation_operator.set_search_space(self.search_space)

        self.current_population = self.get_initial_population()
        self.current_model = self.get_initial_model()

    def get_improved_model(self,
                           previous_model: list[PSIndividual],
                           metric: Metric,
                           amount_of_iterations: int,
                           pRef: PRef) -> list[PSIndividual]:
        algorithm = MPLLR(food_weight=0.3,
                          mu_parameter=self.ps_model_selected_amount,
                          lambda_parameter=self.ps_model_size,
                          metric=metric,
                          mutation_operator=self.ps_mutation_operator,
                          starting_population=previous_model)

        algorithm.set_pRef(pRef)

        termination_criteria = TerminationCriteria.IterationLimit(amount_of_iterations)
        algorithm.run(termination_criteria)
        return algorithm.get_results(quantity_returned=self.ps_model_size)

    def improve_exploitative_model(self):
        metrics = Averager([MeanFitness(), BiVariateANOVALinkage()])
        pRef = self.get_pRef_from_current_population()

        self.current_model = self.get_improved_model(previous_model=self.current_model+self.get_initial_model(),
                                                     metric=metrics,
                                                     amount_of_iterations=6,
                                                     pRef = pRef)


    def improve_population(self):
        sampler = FSSampler(self.search_space,
                            self.current_model)
        new_solutions = [sampler.sample() for _ in range(self.fs_offspring_size)]
        new_solution_individuals = [FSIndividual(sol, self.fitness_function_evaluator._fitness_function(sol))
                                    for sol in new_solutions]
        new_population = self.current_population + new_solution_individuals
        self.current_population = heapq.nlargest(iterable=new_population, n=self.fs_population_size)

    def get_initial_population(self):
        samples = [FullSolution.random(self.search_space) for _ in range(self.fs_population_size)]
        return [FSIndividual(sample, self.fitness_function_evaluator.evaluate(sample)) for sample in samples]

    def get_pRef_from_current_population(self):
        full_solutions = [individual.full_solution for individual in self.current_population]
        fitness_values = [individual.fitness for individual in self.current_population]
        return PRef.from_full_solutions(full_solutions=full_solutions,
                                        fitness_values=fitness_values,
                                        search_space=self.search_space)


    def run(self, termination_criteria: TerminationCriteria):

        iteration = 0

        def should_stop():
            return termination_criteria.met(iterations = iteration,
                                            fs_evaluations = self.fitness_function_evaluator.used_evaluations)

        while not should_stop():
            self.print_current_state()

            self.improve_exploitative_model()
            self.improve_population()

            iteration += 1

    def print_current_state(self):
        print("The current state is ")

        def state_of_population(population):
            fitness_array = np.array([fsi.fitness for fsi in population])
            stats = utils.get_descriptive_stats(fitness_array)
            return (f"\t#samples = {len(fitness_array)}, "
                    f"\n\tmin={stats[0]:.2f}, median={stats[1]:.2f}, max={stats[2]:.2f}, "
                    f"\n\tavg={stats[3]:.2f}, stdev={stats[4]:.2f}")

        def show_model(model: list[PSIndividual]):
            if len(model) == 0:
                print("\tempty")
            else:
                for item in model[:5]:
                    print(f"\t{item.ps}, score = {item.aggregated_score:.3f}")

        print(f"The current pRef is \n{state_of_population(self.current_population)}")
        print("The cutting edge model is")
        show_model(self.current_model)

    def get_results(self, quantity_returned: int) -> list[FSIndividual]:
        return heapq.nlargest(iterable = self.current_population, n=quantity_returned)

    def get_initial_model(self):
        return [Individual(PS.empty(self.search_space))]


def test_train_tracks_EDA(benchmark_problem: BenchmarkProblem):
    print("Initialising the TT EDA")
    eda = TrainTracksEDA(search_space=benchmark_problem.search_space,
                         fitness_function=benchmark_problem.fitness_function,
                         fs_population_size=10000,
                         fs_offspring_size=200,
                         ps_model_size=100,
                         ps_model_mu=20,
                         ps_mutation_operator=MultimodalMutationOperator(0.5))

    eda.run(TerminationCriteria.IterationLimit(24))

    print("The final state is ")
    eda.print_current_state()

    print("Where the final model was")
    for ps_individual in eda.current_model:
        print(f"{benchmark_problem.repr_ps(ps_individual.ps)}")

    fss = eda.get_results(12)
    print("The Top 12 fss are ")
    for fs_individual in fss:
        fs = fs_individual.full_solution
        as_ps = PS.from_FS(fs)
        fitness = eda.fitness_function_evaluator._fitness_function(fs)
        print(f"{benchmark_problem.repr_ps(as_ps)}, with fitness = {fitness}")



