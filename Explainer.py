# Explainer? I barely know her!
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from EvaluatedFS import EvaluatedFS
from EvaluatedPS import EvaluatedPS
from FullSolution import FullSolution
from PRef import PRef
from PS import contains, PS
from PSMetric.LocalPerturbation import BivariateLocalPerturbation, UnivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from PSMetric.SignificantlyHighAverage import SignificantlyHighAverage


class Explainer:
    benchmark_problem: BenchmarkProblem  # used mainly for repr_pr
    ps_catalog: list[EvaluatedPS]
    pRef: PRef

    mean_fitness_metric: MeanFitness
    statistically_high_fitness_metric: SignificantlyHighAverage
    local_importance_metric: UnivariateLocalPerturbation
    local_linkage_metric: BivariateLocalPerturbation


    def __init__(self,
                 benchmark_problem: BenchmarkProblem,
                 ps_catalog: list[EvaluatedPS],
                 pRef: PRef):
        self.benchmark_problem = benchmark_problem
        self.ps_catalog = ps_catalog
        self.pRef = pRef

        self.mean_fitness_metric = MeanFitness()
        self.statistically_high_fitness_metric = SignificantlyHighAverage()
        self.local_importance_metric = UnivariateLocalPerturbation()
        self.local_linkage_metric = BivariateLocalPerturbation()

        for metric in [self.mean_fitness_metric, self.statistically_high_fitness_metric, self.local_importance_metric, self.local_linkage_metric]:
            metric.set_pRef(self.pRef)


    def t_test_for_mean_with_ps(self, ps: PS) -> (float, float):
        return self.statistically_high_fitness_metric.get_p_value_and_sample_mean(ps)


    def get_small_description_of_ps(self, ps: PS) -> str:
        p_value, mean = self.t_test_for_mean_with_ps(ps)
        return f"{self.benchmark_problem.repr_ps(ps)}, mean fitness = {mean:.2f}, p-value = {p_value:e}"

    def local_explanation_of_full_solution(self, full_solution: FullSolution):
        contained_pss = [ps.ps for ps in self.ps_catalog if contains(full_solution, ps.ps)]

        fs_as_ps = PS.from_FS(full_solution)
        print(f"The solution {self.benchmark_problem.repr_ps(fs_as_ps)} contains the following PSs:")
        for ps in contained_pss:
            print("\t" + self.get_small_description_of_ps(ps))


        # local_importances = self.local_importance_metric.get_local_importance_array(fs_as_ps)
        # local_linkages = self.local_linkage_metric.get_local_linkage_table(fs_as_ps)

    def local_explanation_of_ps(self, ps: PS):
        local_importances = self.local_importance_metric.get_local_importance_array(ps)
        local_linkages = self.local_linkage_metric.get_local_linkage_table(ps)

        # TODO find a good way to display them

