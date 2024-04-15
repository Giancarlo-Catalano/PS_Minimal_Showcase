import TerminationCriteria
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from EvaluatedPS import EvaluatedPS
from PRef import PRef
from PSMetric.Averager import Averager
from PSMetric.Linkage import Linkage
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric
from PSMiners.Operators.PSMutationOperator import PSMutationOperator, SinglePointMutation, MultimodalMutationOperator
from PSMiners.Operators.PSSelectionOperator import PSSelectionOperator, TruncationSelection
from PSMiners.AbstractPSMiner import AbstractPSMiner


class MuPlusLambda(AbstractPSMiner):
    mu_parameter: int
    lambda_parameter: int

    offspring_amount: int  # self.lambda_parameter // self.mu_parameter

    def __init__(self,
                 mu_parameter: int,
                 lambda_parameter: int,
                 metric: Metric,
                 mutation_operator: PSMutationOperator,
                 selection_operator: PSSelectionOperator,
                 pRef: PRef,
                 seed_population=None,
                 set_pRef_in_metric = True):
        self.mu_parameter = mu_parameter
        self.lambda_parameter = lambda_parameter
        self.offspring_amount = self.lambda_parameter // self.mu_parameter
        assert (self.lambda_parameter % self.mu_parameter == 0)

        super().__init__(metric=metric,
                         pRef=pRef,
                         mutation_operator=mutation_operator,
                         selection_operator=selection_operator,
                         seed_population=seed_population,
                         set_pRef_in_metric = set_pRef_in_metric)

    def __repr__(self):
        return f"MuPlusLambda(mu={self.mu_parameter}, lambda = {self.lambda_parameter}, mutation = {self.mutation_operator}, selection = {self.selection_operator})"

    def get_initial_population(self):
        return AbstractPSMiner.get_mixed_initial_population(search_space=self.search_space,
                                                            from_uniform=0,
                                                            from_geometric=1,
                                                            from_half_fixed=0,
                                                            population_size=self.lambda_parameter)

    def get_offspring(self, individual: EvaluatedPS) -> list[EvaluatedPS]:
        return [EvaluatedPS(self.mutation_operator.mutated(individual.ps))
                for _ in range(self.offspring_amount)]

    def step(self):
        self.current_population = self.without_duplicates(self.current_population)
        selected_parents = self.selection_operator.select_n(self.mu_parameter, self.current_population)

        children = []
        for parent in selected_parents:
            children.extend(self.get_offspring(parent))

        children = self.evaluate_individuals(children)

        self.current_population = AbstractPSMiner.without_duplicates(selected_parents + children)

    def get_results(self, quantity_returned: int) -> list[EvaluatedPS]:
        return self.get_best_n(n=quantity_returned, population=self.current_population)

    def get_parameters_as_dict(self) -> dict:
        return {"kind": "MPL",
                "mu": self.mu_parameter,
                "lambda": self.lambda_parameter,
                "metric": repr(self.metric.__repr__()),
                "selection": repr(self.selection_operator),
                "mutation": repr(self.mutation_operator)}

    @classmethod
    def with_default_settings(cls, pRef: PRef):
        return cls(mu_parameter=50,
                   lambda_parameter=300,
                   metric=Averager([MeanFitness(), Linkage()]),
                   mutation_operator=MultimodalMutationOperator(0.5, pRef.search_space),
                   selection_operator=TruncationSelection(),
                   pRef=pRef)
