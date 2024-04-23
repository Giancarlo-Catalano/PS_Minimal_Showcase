import csv
import functools
import itertools
import random
from typing import TypeAlias, Iterable, Literal

import numpy as np

from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.Worker import Worker
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem, ExtendedPattern, \
    rota_to_extended_pattern
from DEAP.Testing import get_history_pRef, get_toolbox_for_problem, nsgaii, get_stats_object, \
    report_in_order_of_last_metric
from EvaluatedPS import EvaluatedPS
from PS import PS, STAR
from PSMetric.Atomicity import Atomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from utils import announce


class PSComponent:
    worker: Worker
    chosen_rota_index: int
    chosen_rota_entended: ExtendedPattern


    def __init__(self,
                 worker: Worker,
                 rota_index: int,
                 calendar_length: int):
        self.worker = worker
        self.chosen_rota_index = rota_index
        self.chosen_rota_extended = rota_to_extended_pattern(rota=worker.available_rotas[rota_index],
                                                             calendar_length=calendar_length)


Cohort: TypeAlias = list[PSComponent]


def get_amount_of_shared_skills(cohort: Cohort) -> int:
    if len(cohort) == 0:
        return 0

    skillsets = [component.worker.available_skills for component in cohort]
    common_to_all = set.intersection(*skillsets)
    return len(common_to_all)


def get_hamming_distances(cohort: Cohort) -> list[int]:
    def hamming_distance(component_a: PSComponent, component_b: PSComponent) -> int:
        rota_a = component_a.chosen_rota_extended
        rota_b = component_b.chosen_rota_extended

        return int(np.sum(rota_a != rota_b))

    return [hamming_distance(a, b)
               for a, b in itertools.combinations(cohort, 2)]


def get_average_hamming_distance(cohort: Cohort) -> float:
    distances = get_hamming_distances(cohort)
    return np.average(distances)


def get_working_day_count_max_difference(cohort: Cohort) -> int:
    workcounts = [component.chosen_rota_extended.sum(dtype=int) for component in cohort]
    return max(workcounts) - min(workcounts)

def get_skill_amount_max_difference(cohort: Cohort) -> int:
    amounts = [len(component.worker.available_skills) for component in cohort]
    return max(amounts) - min(amounts)


def get_local_fitness(cohort: Cohort):
    present_skills = set(skill for component in cohort
                         for skill in component.worker.available_skills)
    #TODO



class BTProblemPatternDetector:
    bt_problem: BTProblem


    def __init__(self, bt_problem: BTProblem):
        self.bt_problem = bt_problem


    def ps_to_cohort(self, ps: PS) -> Cohort:
        def fixed_var_to_PSComponent(var: int, val: int) -> PSComponent:
            worker = self.bt_problem.workers[var]
            return PSComponent(worker, val, calendar_length=self.bt_problem.calendar_length)
        return [fixed_var_to_PSComponent(var, val)
                for var, val in enumerate(ps.values)
                if val != STAR]

    def random_cohort(self, size: int) -> Cohort:
        def random_PSComponent() -> PSComponent:
            random_worker: Worker = random.choice(self.bt_problem.workers)
            rota_index = random.randrange(len(random_worker.available_rotas))
            return PSComponent(random_worker, rota_index, self.bt_problem.calendar_length)

        return [random_PSComponent() for _ in range(size)]


    def get_row_for_cohort(self, cohort: Cohort, control: bool) -> dict:
        return {"size": len(cohort),
                "q_shared_skills": get_amount_of_shared_skills(cohort),
                "hamming_avg": get_average_hamming_distance(cohort),
                "workcount_range": get_working_day_count_max_difference(cohort),
                "skill_q_range": get_skill_amount_max_difference(cohort),
                "control": control}


    def get_rows_from_pss(self, pss: Iterable[PS],
                          control_amount: int) -> list[dict]:
        given_cohorts = [self.ps_to_cohort(ps) for ps in pss if ps.fixed_count() > 1]
        control_cohorts = [self.random_cohort(size=len(random.choice(given_cohorts)))
                           for _ in range(control_amount)]

        return list(itertools.chain(
            map(functools.partial(self.get_row_for_cohort, control=False), given_cohorts),
            map(functools.partial(self.get_row_for_cohort, control=True), control_cohorts)))


    def evaluated_pss_into_csv(self, csv_file_name: str,
                               pss: list[PS]):
        with announce("Converting the cohorts into rows"):
            rows = self.get_rows_from_pss(pss, len(pss*6))
        headers = rows[0].keys()
        with open(csv_file_name, "w+", newline="") as file:
            csv_writer = csv.DictWriter(file, fieldnames=headers)
            csv_writer.writeheader()
            csv_writer.writerows(rows)



def test_and_produce_patterns(benchmark_problem: BTProblem,
                              csv_file_name: str,
                              pRef_size: int,
                              method: Literal["uniform", "GA", "SA"], ):
    pRef = get_history_pRef(benchmark_problem=benchmark_problem,
                            which_algorithm = method,
                            sample_size = pRef_size)

    metrics = [Simplicity(), MeanFitness(), Atomicity()]
    with announce("Running the PS_mining algorithm"):
        toolbox = get_toolbox_for_problem(benchmark_problem,
                                          metrics, use_experimental_niching=True,
                                          pRef = pRef)
        final_population, logbook = nsgaii(toolbox=toolbox,
                                           mu = 300,
                                           cxpb=0.5,
                                           mutpb=1/benchmark_problem.search_space.amount_of_parameters,
                                           ngen=1000,
                                           stats=get_stats_object())
    print("The last population is ")
    report_in_order_of_last_metric(final_population, benchmark_problem)

    detector = BTProblemPatternDetector(benchmark_problem)
    with announce(f"Analysing data and writing to file {csv_file_name}"):
        detector.evaluated_pss_into_csv(csv_file_name, final_population)






