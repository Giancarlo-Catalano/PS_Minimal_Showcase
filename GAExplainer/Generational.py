from typing import Callable, TypeAlias

import numpy as np

from BaselineApproaches.FullSolutionGA import FullSolutionGA
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from EDA.FSIndividual import FSIndividual
from PRef import PRef
from PS import PS
from PSMetric.Averager import Averager
from PSMetric.BivariateANOVALinkage import BivariateANOVALinkage
from PSMetric.LocalPerturbation import UnivariateLocalPerturbation, BivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from PSMiners.MuPlusLambda.MPLLR import MPLLR
from PSMiners.Operators.PSMutationOperator import MultimodalMutationOperator
from PSMiners.Operators.PSSelectionOperator import TruncationSelection
from PSMiners.PSMiner import PSMiner
from TerminationCriteria import TerminationCriteria, IterationLimit, PSEvaluationLimit

PSMinerFactory: TypeAlias = Callable[[PRef], PSMiner]
PSCatalog: TypeAlias = list[PS]


class Generational:
    iterative_algorithm: FullSolutionGA
    ps_miner_factory: Callable[[], PSMiner]

    pRefs: list[PRef]
    ps_catalogs: list[PSCatalog]
    univariate_local_perturbation_metric: UnivariateLocalPerturbation
    bivariate_local_perturbation_metric: BivariateLocalPerturbation

    bivariate_global_linkage_metric: BivariateANOVALinkage

    def __init__(self,
                 iterative_algorithm: FullSolutionGA,
                 ps_miner_factory: PSMinerFactory):
        self.iterative_algorithm = iterative_algorithm
        self.ps_miner_factory = ps_miner_factory

        self.pRefs = [self.get_pRef_from_fs_population(self.iterative_algorithm.current_population)]
        self.ps_catalogs = [[]]
        self.univariate_local_perturbation_metric = UnivariateLocalPerturbation()
        self.bivariate_local_perturbation_metric = BivariateLocalPerturbation()
        self.bivariate_global_linkage_metric = BivariateANOVALinkage()

    def get_pRef_from_fs_population(self, fs_population: list[FSIndividual]) -> PRef:
        full_solutions = [ind.full_solution for ind in fs_population]
        fitness_values = [ind.fitness for ind in fs_population]
        return PRef.from_full_solutions(full_solutions=full_solutions,
                                        fitness_values=fitness_values,
                                        search_space=self.iterative_algorithm.search_space)

    def get_PS_catalog_from_pRef(self, pRef: PRef, ps_miner_termination_criterion: TerminationCriteria) -> PSCatalog:
        ps_miner: PSMiner = self.ps_miner_factory(pRef)
        ps_miner.run(ps_miner_termination_criterion)

        individuals = ps_miner.get_results(20)  # arbitrary
        return [ind.ps for ind in individuals]

    def step(self, ps_miner_termination_criterion: TerminationCriteria):
        self.iterative_algorithm.step()
        new_pRef = self.get_pRef_from_fs_population(self.iterative_algorithm.current_population)
        self.pRefs.append(new_pRef)
        new_catalog = self.get_PS_catalog_from_pRef(new_pRef, ps_miner_termination_criterion)
        self.ps_catalogs.append(new_catalog)

    def run(self,
            fs_ga_termination_criterion: TerminationCriteria,
            ps_miner_termination_criterion: TerminationCriteria):
        iterations = 0

        def should_stop():
            return fs_ga_termination_criterion.met(iterations=iterations,
                                                   fs_evaluations=self.iterative_algorithm.evaluator.used_evaluations)

        while not should_stop():
            self.step(ps_miner_termination_criterion)

    def get_results(self, quantity_returned: int) -> list[FSIndividual]:
        return self.iterative_algorithm.get_results(quantity_returned)


    def get_local_interaction_table_for_ps(self, ps: PS) -> np.ndarray:
        self.bivariate_local_perturbation_metric.set_pRef(self.pRefs[-1])
        return self.bivariate_local_perturbation_metric.get_local_linkage_table(ps)


def test_generational(benchmark_problem: BenchmarkProblem):
    ga = FullSolutionGA(search_space=benchmark_problem.search_space,
                        mutation_rate=1 / benchmark_problem.search_space.amount_of_parameters,
                        crossover_rate=0.5,
                        elite_size=3,
                        tournament_size=3,
                        population_size=10000,
                        fitness_function=benchmark_problem.fitness_function)

    def ps_miner_factory(pRef: PRef) -> PSMiner:
        metrics = Averager([MeanFitness(), BivariateANOVALinkage()])
        metrics.set_pRef(pRef)
        return MPLLR(mu_parameter=50,
                     lambda_parameter=300,
                     pRef=pRef,
                     food_weight=0.5,
                     metric=metrics,
                     mutation_operator=MultimodalMutationOperator(0.5, search_space=pRef.search_space),
                     selection_operator=TruncationSelection())



    print("Starting the main algorithm")
    algorithm = Generational(iterative_algorithm=ga,
                             ps_miner_factory=ps_miner_factory)


    print("Running the main algorithm")
    algorithm.run(fs_ga_termination_criterion=IterationLimit(20),
                  ps_miner_termination_criterion=PSEvaluationLimit(15000))



    print("Generational explainer has finished, you should be debugging")
