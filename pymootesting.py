import random

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from EvaluatedPS import EvaluatedPS
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Metric import Metric
from PSMetric.Simplicity import Simplicity
from utils import announce


def convert_pymoo_individual_to_ps(pymoo_individual) -> PS:
    """
    I think that internally they are numpy arrays..
    """
    return PS(pymoo_individual)

class PyMooPSMiningProblem(ElementwiseProblem):
    original_problem: BenchmarkProblem
    metrics: list[Metric]

    def __init__(self, benchmark_problem: BenchmarkProblem,
                 metrics: list[Metric]):
        self.original_problem = benchmark_problem
        self.metrics = metrics
        super().__init__(n_var=self.original_problem.search_space.amount_of_parameters,
                         n_obj=len(self.metrics),
                         n_ieq_constr=0,
                         xl=np.array([-1 for card in self.original_problem.search_space.cardinalities]),
                         xu=np.array([card-1 for card in self.original_problem.search_space.cardinalities]),
                         vtype=int)  # this might change

    def _evaluate(self, x, out, *args, **kwargs):
        ps = convert_pymoo_individual_to_ps(x)
        out["F"] = np.column_stack([metric.get_single_score(ps) for metric in self.metrics])


class PSSamplingForPyMoo(IntegerRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        n, (xl, xu) = problem.n_var, problem.bounds()
        result = np.array([-1 for _ in range(n)], dtype=int)
        chance_of_success = 0.79
        while random.random() < chance_of_success:
            var_index = random.randrange(n)
            value = random.randrange(xu[var_index]+1)
            result[var_index] = value
        return np.column_stack(result)

def solve_with_pymoo(benchmark_problem: BenchmarkProblem,
                     metrics: list[Metric],
                     pRef: PRef):
    for metric in metrics:
        metric.set_pRef(pRef)
    pymooo_problem = PyMooPSMiningProblem(benchmark_problem=benchmark_problem,
                                          metrics=metrics)
    ref_dirs = get_reference_directions("das-dennis", len(metrics), n_partitions=12)

    algorithm = NSGA3(pop_size=300,
                      ref_dirs=ref_dirs,
                      sampling=PSSamplingForPyMoo(),
                      crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                      eliminate_duplicates=True
                      )

    termination = ('n_gen', 40)

    with announce("Running pymoo"):
        res = minimize(problem=pymooo_problem,
                       algorithm=algorithm,
                       termination=termination,
                       verbose=True)

    final_solutions = [EvaluatedPS(PS(values)) for values in res.X]
    for sol, fitness in zip(final_solutions, res.F):
        sol.aggregated_fitness = fitness

    print("The final solutions are ")
    for e_ps in final_solutions:
        print(f"{benchmark_problem.repr_ps(e_ps.ps)}, metrics = {e_ps.aggregated_score}")



def test_run_with_pymoo(benchmark_problem: BenchmarkProblem,
                        sample_size = 10000):
    metrics = [Simplicity(), MeanFitness(), Atomicity()]
    pRef = benchmark_problem.get_reference_population(sample_size)
    solve_with_pymoo(benchmark_problem,
                     pRef = pRef,
                     metrics = metrics)