import heapq
from typing import Optional, TypeAlias

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FSEvaluator import FSEvaluator
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.Linkage import Linkage
from PSMetric.LocalPerturbation import BivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric
from PSMetric.Simplicity import Simplicity
from EvaluatedPS import EvaluatedPS
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria, PSEvaluationLimit
from get_init import just_empty
from get_local import specialisations
from selection import truncation_selection
from utils import announce

Population: TypeAlias = list[EvaluatedPS]
GetInitType: TypeAlias = [[PRef, Optional[int]], list[PS]]
GetLocalType: TypeAlias = [[PS, SearchSpace], list[PS]]
SelectionType: TypeAlias = [[list[EvaluatedPS], int], list[EvaluatedPS]]

class PSMiner:
    """This class is the PS miner, which outputs a PS catalog when used right"""
    """There are many parts that can be modified, and these were tested in the paper, 
    but you should probably just use with_default_settings as a constructor"""

    metrics: list[Metric]  # usually they are simplicity, mean_fitness, atomicity
    population_size: int
    get_init: GetInitType  # generates the initial population
    get_local: GetLocalType  # generates the offspring of a selected ps

    pRef: PRef  # the PRef which remains constant throughout the process
    selection: SelectionType  #  the selection operator

    current_population: list[EvaluatedPS]
    archive: set[EvaluatedPS]  # the archive, which will contain all the selected PSs

    used_evaluations: int  # counts how many F_\psi evaluations have happened

    def __init__(self,
                 pRef: PRef,
                 metrics: list[Metric],
                 get_init: GetInitType,
                 get_local: GetLocalType,
                 population_size: int,
                 selection: SelectionType):
        self.used_evaluations = 0

        self.pRef = pRef
        self.metrics = metrics

        for metric in self.metrics:
            metric.set_pRef(self.pRef)

        self.get_init = get_init
        self.get_local = get_local
        self.selection = selection
        self.population_size = population_size

        self.current_population = [EvaluatedPS(ps) for ps in self.get_init(self.pRef, quantity=self.population_size)]
        self.current_population = self.evaluate_individuals(self.current_population)  # experimental
        self.archive = set()



    def __repr__(self):
        return f"PSMiner(population_size = {self.population_size})"

    @property
    def search_space(self):
        return self.pRef.search_space

    def with_aggregated_scores(self, population: list[EvaluatedPS]) -> list[EvaluatedPS]:
        """
        This is kinda the fitness function of PSs, where we
         - remap every metric between individuals, to be in range[0, 1]
         - average the metrics within individuals, to have a single value

         Note that the final fitnesses are RELATIVE to the population, which is why the algorithm is quite slow
        :param population: the population, where ALL of the metrics are assumed to have been calculated
        :return: population: the same population, but now .aggregated_score is a valid value
        """
        metric_matrix = np.array([ind.metric_scores for ind in population])
        for column in range(metric_matrix.shape[1]):
            metric_matrix[:, column] = utils.remap_array_in_zero_one(metric_matrix[:, column])

        averages = np.average(metric_matrix, axis=1)
        for individual, score in zip(population, averages):
            individual.aggregated_score = score

        return population

    def step(self):
        """ The contents of the main loop"""

        self.current_population = self.without_duplicates(self.current_population)

        # aggregate the various objectives into a single score
        self.current_population = self.with_aggregated_scores(self.current_population)
        # truncate population
        self.current_population = self.top(n=self.population_size, population=self.current_population)

        # select parents
        parents = self.selection(self.current_population, self.population_size // 3)
        parents = self.without_duplicates(parents)

        # get offspring
        children = [EvaluatedPS(child) for parent in parents for child in parent.ps.specialisations(self.search_space)]

        # add selected individuals to archive
        self.archive.update(parents)

        self.current_population.extend(children)

        # remove from population the individuals that appear in the archive (including the parents]
        self.current_population = [ind for ind in self.current_population if ind not in self.archive]

        self.current_population = self.evaluate_individuals(self.current_population)

    def evaluate_individuals(self, newborns: Population) -> Population:
        """
        Calculates the metrics for each individual, but this is not the true fitness function!
        These metrics are ABSOLUTE, ie they are not relative to the population, although they are relative to the PRef.
        :param newborns: the individuals to be evaluated
        :return: the same individuals as the input, but now .metric_scores will be valid
        """
        for individual in newborns:
            if individual.metric_scores is None:  # avoid recalculating if already valid
                individual.metric_scores = [metric.get_single_score(individual.ps) for metric in self.metrics]
                self.used_evaluations += 1
        return newborns

    def run(self, termination_criteria: TerminationCriteria):
        """ Executes the main loop, with the termination criterion usually being an evaluation budget"""
        iterations = 0

        def should_terminate():
            return termination_criteria.met(iterations=iterations,
                                            ps_evaluations=self.used_evaluations) or len(self.current_population) == 0

        while not should_terminate():
            self.step()
            iterations +=1

    def get_results(self, quantity_returned: int) -> list[EvaluatedPS]:
        """
        This is the only way you should get the result out of this!!
        This method will evaluate the archive and return the top n
        Note that the evaluation is relative to the archive, so the aggregated scores are recalculated
        :param quantity_returned:
        :return: The best PSs in the archive, of the quantity specified
        """
        evaluated_archive = self.with_aggregated_scores(list(self.archive))
        return self.top(n=quantity_returned, population=evaluated_archive)

    @staticmethod
    def top(n: int, population: Population) -> Population:
        """ Same as in the paper, very straightforward"""
        return heapq.nlargest(n=n, iterable=population)

    @staticmethod
    def without_duplicates(population: Population) -> Population:
        return list(set(population))
    @classmethod
    def with_default_settings(cls, pRef: PRef):
        """ atomicity can be measured in many many ways, and the paper suggest an approach that I've improved over time"""
        """The function defined in the paper uses Atomicity(), but you should also try:
            - Linkage(): faster
            - BivariateLocalPerturbation(): much more accurate, but sloooow
            - BivariateANOVALinkage(): slow but more mathematically sound
            
        """
        return cls(population_size=150,
                   pRef=pRef,
                   metrics=[Simplicity(), MeanFitness(), Linkage()],
                   get_init=just_empty,
                   get_local=specialisations,
                   selection=truncation_selection)


    @classmethod
    def test_with_problem(cls, benchmark_problem: BenchmarkProblem):
        """ This method is mainly for debug, but you might find it useful too"""
        evaluator = FSEvaluator(benchmark_problem.fitness_function)

        with announce("Gathering pRef"):
            pRef = evaluator.generate_pRef_from_search_space(search_space=benchmark_problem.search_space,
                                                             amount_of_samples=10000)
        with announce("Running the algorithm"):
            algorithm: PSMiner = cls.with_default_settings(pRef)
            algorithm.run(PSEvaluationLimit(15000))

        print("The best results are")
        best = algorithm.get_results(12)
        for item in best:
            print(item)