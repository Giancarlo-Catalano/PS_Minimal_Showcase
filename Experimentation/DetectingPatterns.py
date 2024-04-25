import csv
import functools
import itertools
import json
import random
from typing import TypeAlias, Iterable, Literal

import numpy as np
from matplotlib import pyplot as plt

from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.Worker import Worker
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem, ExtendedPattern, \
    rota_to_extended_pattern
from DEAP.Testing import get_history_pRef, get_toolbox_for_problem, nsgaii, get_stats_object, \
    report_in_order_of_last_metric
from EvaluatedPS import EvaluatedPS
from PRef import plot_solutions_in_pRef
from PS import PS, STAR
from PSMetric.Atomicity import Atomicity
from PSMetric.LocalPerturbation import BivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from custom_types import JSON
from utils import announce
import seaborn as sns
import pandas as pd
from statsmodels.graphics.boxplots import violinplot

class CohortMember:
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

    def to_json(self) -> JSON:
        return {"worker": self.worker.to_json(),
                "chosen_rota": int(self.chosen_rota_index)}

    @classmethod
    def from_json(cls, element: JSON):
        chosen_rota = int(element["chosen_rota"])
        calendar_length = 91  # TODO fetch this from somewhere else
        worker = Worker.from_json(element["worker"])
        return cls(worker=worker, rota_index=chosen_rota, calendar_length=calendar_length)



Cohort: TypeAlias = list[CohortMember]


def cohort_to_json(cohort: Cohort) -> JSON:
    return [member.to_json() for member in cohort]
def get_amount_of_shared_skills(cohort: Cohort) -> int:
    if len(cohort) == 0:
        return 0

    skillsets = [component.worker.available_skills for component in cohort]
    common_to_all = set.intersection(*skillsets)
    return len(common_to_all)


def get_hamming_distances(cohort: Cohort) -> list[int]:
    def hamming_distance(component_a: CohortMember, component_b: CohortMember) -> int:
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


def get_skill_variation(cohort: Cohort) -> float:
    all_skills = set(skill for component in cohort
                          for skill in component.worker.available_skills)
    sum_of_available_skills = sum(len(component.worker.available_skills) for component in cohort)
    return len(all_skills) / sum_of_available_skills




class BTProblemPatternDetector:
    bt_problem: BTProblem


    def __init__(self, bt_problem: BTProblem):
        self.bt_problem = bt_problem


    def ps_to_cohort(self, ps: PS) -> Cohort:
        def fixed_var_to_cohort_member(var: int, val: int) -> CohortMember:
            worker = self.bt_problem.workers[var]
            return CohortMember(worker, val, calendar_length=self.bt_problem.calendar_length)
        return [fixed_var_to_cohort_member(var, val)
                for var, val in enumerate(ps.values)
                if val != STAR]


    def get_distribution_of_skills(self) -> dict:
        distribution = {skill: 0 for skill in self.bt_problem.all_skills}
        for worker in self.bt_problem.workers:
            for skill in worker.available_skills:
                distribution[skill] = distribution[skill]+1

        return distribution

    def random_cohort(self, size: int) -> Cohort:
        def random_PSComponent() -> CohortMember:
            random_worker: Worker = random.choice(self.bt_problem.workers)
            rota_index = random.randrange(len(random_worker.available_rotas))
            return CohortMember(random_worker, rota_index, self.bt_problem.calendar_length)

        return [random_PSComponent() for _ in range(size)]


    def get_row_for_cohort(self, cohort: Cohort, control: bool) -> dict:
        return {"size": len(cohort),
                "q_shared_skills": get_amount_of_shared_skills(cohort),
                "hamming_avg": get_average_hamming_distance(cohort),
                "workcount_range": get_working_day_count_max_difference(cohort),
                "skill_q_range": get_skill_amount_max_difference(cohort),
                "skill_variation": get_skill_variation(cohort),
                "control": control}


    def get_rows_from_cohorts(self, cohorts: list[Cohort], is_control = False) -> list[dict]:
        return [self.get_row_for_cohort(cohort, is_control) for cohort in cohorts]


    def cohorts_into_csv(self, csv_file_name: str,
                         cohorts: list[Cohort],
                         remove_trivial_ps = True,
                         verbose = False):
        if remove_trivial_ps:
            cohorts = [cohort for cohort in cohorts if len(cohort) > 1]
        with announce("Converting the cohorts into rows", verbose):
            rows = self.get_rows_from_cohorts(cohorts)
        headers = rows[0].keys()
        with open(csv_file_name, "w+", newline="") as file:
            csv_writer = csv.DictWriter(file, fieldnames=headers)
            csv_writer.writeheader()
            csv_writer.writerows(rows)



def test_and_produce_patterns(benchmark_problem: BTProblem,
                              csv_file_name: str,
                              pRef_size: int,
                              method: Literal["uniform", "GA", "SA"],
                              verbose = False):
    pRef = get_history_pRef(benchmark_problem=benchmark_problem,
                            which_algorithm = method,
                            sample_size = pRef_size,
                            verbose=verbose)

    if verbose:
        print("Here is the pRef, in case you were curious")
        plot_solutions_in_pRef(pRef)

    metrics = [Simplicity(), MeanFitness(), BivariateLocalPerturbation()]
    with announce("Running the PS_mining algorithm", verbose = verbose):
        toolbox = get_toolbox_for_problem(benchmark_problem,
                                          metrics, use_experimental_niching=True,
                                          pRef = pRef)
        final_population, logbook = nsgaii(toolbox=toolbox,
                                           mu =200,
                                           cxpb=0.5,
                                           mutpb=1/benchmark_problem.search_space.amount_of_parameters,
                                           ngen=2,
                                           stats=get_stats_object(),
                                           verbose=verbose)
    if verbose:
        print("The last population is ")
        report_in_order_of_last_metric(final_population, benchmark_problem)

    def get_table_for_individual(ps:PS):
        return metrics[-1].get_local_linkage_table(ps)

    detector = BTProblemPatternDetector(benchmark_problem)
    cohorts = [detector.ps_to_cohort(ps) for ps in final_population]
    with announce(f"Analysing data and writing to file {csv_file_name}", verbose):
        detector.cohorts_into_csv(csv_file_name, cohorts)

    if verbose:
        print("Either way, the cohorts were:>>")

    cohorts_as_json = [cohort_to_json(cohort) for cohort in cohorts]
    print(json.dumps(cohorts_as_json))


def mine_cohorts_from_problem(benchmark_problem: BTProblem,
                              method: Literal["uniform", "GA", "SA"],
                              pRef_size: int,
                              nsga_pop_size: int,
                              verbose=True):
    pRef = get_history_pRef(benchmark_problem=benchmark_problem,
                            which_algorithm = method,
                            sample_size = pRef_size,
                            verbose=False)

    if verbose:
        plot_solutions_in_pRef(pRef)

    metrics = [Simplicity(), MeanFitness(), BivariateLocalPerturbation()]
    with announce("Running the PS_mining algorithm", verbose = verbose):
        toolbox = get_toolbox_for_problem(benchmark_problem,
                                          metrics, use_experimental_niching=True,
                                          pRef = pRef)
        final_population, logbook = nsgaii(toolbox=toolbox,
                                           mu =nsga_pop_size,
                                           cxpb=0.5,
                                           mutpb=1/benchmark_problem.search_space.amount_of_parameters,
                                           ngen=600,
                                           stats=get_stats_object(),
                                           verbose=verbose)

    detector = BTProblemPatternDetector(benchmark_problem)
    return [detector.ps_to_cohort(ps) for ps in final_population]



def json_to_cohorts(json_file: str) -> list[Cohort]:
    with open(json_file, "r") as file:
        data = json.load(file)
    return [[CohortMember.from_json(element) for element in cohort]
            for cohort in data]


def cohorts_to_json(cohorts: list[Cohort]) -> JSON:
    return [cohort_to_json(cohort) for cohort in cohorts]



def plot_nicely_old(input_csv_file: str):


    # Load your CSV file
    df = pd.read_csv(input_csv_file)

    # Define the plot
    plt.figure(figsize=(12, 6))

    # Create the violin plot
    fig, ax = plt.subplots()

    clique_sizes = sorted(pd.unique(df["clique_size"]))
    for clique_size in clique_sizes:
        where_clique_matches = df["clique_size"] == clique_size
        control_data = df[where_clique_matches & df["is_control"] == True]["skills_diversity"]
        real_data = df[where_clique_matches & df["is_control"] == False]["skills_diversity"]

        violinplot([control_data], positions=[0], show_boxplot=False, side='left', ax=ax, plot_opts={'violin_fc': 'C0'})
        violinplot([real_data], positions=[0], show_boxplot=False, side='right', ax=ax, plot_opts={'violin_fc':'C1'})

    # Customize the plot
    plt.title('Asymmetric Violin Plot: Skills Diversity by Clique Size and Control')
    plt.xlabel('Clique Size')
    plt.ylabel('Skills Diversity (%)')
    plt.legend(title='Control Group', loc='upper left')

    # Show the plot
    plt.show()


def plot_nicely(input_csv_file: str):
    # Load your CSV file
    df = pd.read_csv(input_csv_file)

    sns.set(style="whitegrid", palette="pastel", color_codes=True)

    f, ax = plt.subplots(figsize=(8, 8))

    sns.violinplot(x="size", y="skill_variation", hue="control", data=df,
                   palette={True: "b", False: "y"})
    sns.despine(left=True)

    f.suptitle('Skill diversity analysis', fontsize=18, fontweight='bold')
    ax.set_xlabel("clique size",size = 16,alpha=0.7)
    ax.set_ylabel("Skill diversity",size = 16,alpha=0.7)
    plt.legend(loc='upper left')

    plt.show()



