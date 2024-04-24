import csv
import functools
import itertools
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
from PS import PS, STAR
from PSMetric.Atomicity import Atomicity
from PSMetric.LocalPerturbation import BivariateLocalPerturbation
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from utils import announce
import seaborn as sns
import pandas as pd
from statsmodels.graphics.boxplots import violinplot

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
                "skill_variation": get_skill_variation(cohort),
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

    metrics = [Simplicity(), MeanFitness(), BivariateLocalPerturbation()]
    with announce("Running the PS_mining algorithm"):
        toolbox = get_toolbox_for_problem(benchmark_problem,
                                          metrics, use_experimental_niching=True,
                                          pRef = pRef)
        final_population, logbook = nsgaii(toolbox=toolbox,
                                           mu =200,
                                           cxpb=0.5,
                                           mutpb=1/benchmark_problem.search_space.amount_of_parameters,
                                           ngen=100,
                                           stats=get_stats_object())
    print("The last population is ")
    report_in_order_of_last_metric(final_population, benchmark_problem)

    def get_table_for_individual(ps:PS):
        return metrics[-1].get_local_linkage_table(ps)

    detector = BTProblemPatternDetector(benchmark_problem)
    with announce(f"Analysing data and writing to file {csv_file_name}"):
        detector.evaluated_pss_into_csv(csv_file_name, final_population)






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

    sns.violinplot(x="clique_size", y="skills_diversity", hue="is_control", data=df,
                   palette={True: "b", False: "y"})
    sns.despine(left=True)

    f.suptitle('Skill diversity analysis', fontsize=18, fontweight='bold')
    ax.set_xlabel("clique size",size = 16,alpha=0.7)
    ax.set_ylabel("Skill diversity",size = 16,alpha=0.7)
    plt.legend(loc='upper left')

    plt.show()



