from typing import Callable, TypeAlias

import utils
from BaselineApproaches.FullSolutionGA import FullSolutionGA
from EDA.FSEvaluator import FSEvaluator
from EDA.FSIndividual import FSIndividual
from FullSolution import FullSolution
from PRef import PRef
from PS import PS
from PSMetric.BivariateANOVALinkage import BivariateANOVALinkage
from PSMetric.LocalPerturbation import UnivariateLocalPerturbation, BivariateLocalPerturbation
from PSMiners.PSMiner import PSMiner
from SearchSpace import SearchSpace
from TerminationCriteria import TerminationCriteria

PSMinerFactory: TypeAlias = Callable[[PRef], PSMiner]
PSCatalog: TypeAlias = list[PS]
class Generational:
    iterative_algorithm: FullSolutionGA
    ps_miner_factory: Callable[[], PSMiner]

    search_space: SearchSpace
    fs_evaluator: FSEvaluator


    pRefs: list[PRef]
    ps_catalogs: list[PSCatalog]
    univariate_local_perturbation_metric: UnivariateLocalPerturbation
    bivariate_local_perturbation_metric: BivariateLocalPerturbation

    bivariate_global_linkage_metric: BivariateANOVALinkage






    def __init__(self,
                 iterative_algorithm: FullSolutionGA,
                 ps_miner_factory: PSMinerFactory,
                 search_space: SearchSpace,
                 fitness_function: Callable[[FullSolution], float]):
        self.iterative_algorithm = iterative_algorithm
        self.ps_miner_factory = ps_miner_factory
        self.search_space = search_space
        self.fs_evaluator = FSEvaluator(fitness_function)


        self.pRefs = [self.get_pRef_from_fs_population(self.iterative_algorithm.current_population)]
        self.ps_catalogs = []
        self.univariate_local_perturbation_metric = UnivariateLocalPerturbation()
        self.bivariate_local_perturbation_metric = BivariateLocalPerturbation()
        self.bivariate_global_linkage_metric = BivariateANOVALinkage()

    def get_pRef_from_fs_population(self, fs_population: list[FSIndividual]) -> PRef:
        full_solutions = [ind.full_solution for ind in fs_population]
        fitness_values = [ind.fitness for ind in fs_population]
        return PRef.from_full_solutions(full_solutions=full_solutions,
                                        fitness_values=fitness_values,
                                        search_space=self.search_space)

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
            return fs_ga_termination_criterion.met(iterations = iterations,
                                                   fs_evaluations = self.iterative_algorithm.evaluator.used_evaluations)

        while not should_stop():
            self.step(ps_miner_termination_criterion)




    def get_results(self, quantity_returned: int) -> list[FSIndividual]:
        return self.iterative_algorithm.get_results(quantity_returned)






