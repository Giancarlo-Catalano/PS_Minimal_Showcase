import plotly.express as px
import csv
import itertools
import json
import os
import random
from typing import TypeAlias, Iterable, Literal

import numpy as np
import shap
from deap.tools import Logbook
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.Worker import Worker
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import ExtendedPattern, \
    rota_to_extended_pattern
from PSMiners.DEAP.deap_utils import get_toolbox_for_problem, nsga, get_stats_object, \
    report_in_order_of_last_metric
from Core.PRef import plot_solutions_in_pRef
from Core.PS import PS, STAR
from Core.PSMetric.Atomicity import Atomicity
from Core.PSMetric.MeanFitness import MeanFitness
from Core.PSMetric.Simplicity import Simplicity
from Core.custom_types import JSON
from PSMiners.Mining import get_history_pRef, load_ps
from utils import announce
import seaborn as sns
import pandas as pd
from xgboost import XGBClassifier


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


    def get_amount_of_skills(self) -> int:
        return len(self.worker.available_skills)

    def get_amount_of_working_hours(self) -> int:
        return int(np.sum(self.chosen_rota_extended))


    def get_amount_of_choices(self) -> int:
        return len(self.worker.available_rotas)



Cohort: TypeAlias = list[CohortMember]

def cohort_to_json(cohort: Cohort) -> JSON:
    return [member.to_json() for member in cohort]


# utilities for analysing variables

def get_mean_error(values: Iterable) -> float:
    if len(values) < 1:
        raise ValueError
    mean = np.average(values)
    return sum(abs(x - mean) for x in values) / len(values)

def get_max_difference(values: Iterable) -> float:
    return max(values) - min(values)


def get_min_difference(values: Iterable) -> float:
    to_check = np.array(sorted(values))
    differences = to_check[1:] - to_check[:-1]
    return min(differences)


def get_statistical_info_about_iterable(values: Iterable, var_name: str) -> dict:
    return {f"{var_name}_mean": np.average(values),
            f"{var_name}_mean_error": get_mean_error(values),
            f"{var_name}_min_diff": get_min_difference(values),
            f"{var_name}_max_diff": get_max_difference(values)}


def get_amount_of_shared_skills(cohort: Cohort) -> int:
    if len(cohort) == 0:
        return 0

    skillsets = [component.worker.available_skills for component in cohort]
    common_to_all = set.intersection(*skillsets)
    return len(common_to_all)


def get_skill_variation(cohort: Cohort) -> float:
    all_skills = set(skill for component in cohort
                          for skill in component.worker.available_skills)
    sum_of_available_skills = sum(len(component.worker.available_skills) for component in cohort)
    return len(all_skills) / sum_of_available_skills


def get_hamming_distances(cohort: Cohort) -> list[int]:

    def hamming_distance(component_a: CohortMember, component_b: CohortMember) -> int:
        rota_a = component_a.chosen_rota_extended
        rota_b = component_b.chosen_rota_extended

        return int(np.sum(rota_a != rota_b))

    if len(cohort) == 2:  # makes my life a lot easier for data analysis
        distance = hamming_distance(cohort[0], cohort[1])
        return [distance, distance]

    return [hamming_distance(a, b)
               for a, b in itertools.combinations(cohort, 2)]


def get_ranges_in_weekdays(cohort: Cohort) -> np.ndarray:
    total_pattern: np.ndarray = sum(member.chosen_rota_extended for member in cohort)
    total_pattern = total_pattern.reshape((-1, 7))

    def range_score(min_amount, max_amount):
        if max_amount == 0:
            return 0
        return (max_amount - min_amount) / max_amount
    def range_for_column(column_index: int) -> float:
        values = total_pattern[:, column_index]
        min_value = min(values)
        max_values = max(values)
        return range_score(min_value, max_values)

    return np.array([range_for_column(i) for i in range(7)])


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


    def ps_file_to_cohort_file(self, ps_file: str, cohort_file: str, verbose: bool):
        pss = load_ps(ps_file)
        cohorts = [self.ps_to_cohort(ps.ps) for ps in pss]
        with announce(f"Writing the cohorts ({len(cohorts)} onto the file", verbose):
            utils.make_folder_if_not_present(cohort_file)
            cohort_data = cohorts_to_json(cohorts)
            with open(cohort_file, "w+") as file:
                json.dump(cohort_data, file)


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


    def generate_matching_random_cohorts(self, reference_cohorts: list[Cohort], amount_to_generate: int) -> list[Cohort]:
        def pick_size() -> int:
            random_cohort = random.choice(reference_cohorts)
            return len(random_cohort)

        return [self.random_cohort(pick_size()) for _ in range(amount_to_generate)]


    def get_row_for_cohort(self, cohort: Cohort, control: bool) -> dict:
        skill_amounts = [member.get_amount_of_skills() for member in cohort]
        hours_amounts = [member.get_amount_of_working_hours() for member in cohort]
        choices_amounts = [member.get_amount_of_choices() for member in cohort]
        hamming_distances = get_hamming_distances(cohort)
        weekdays = get_ranges_in_weekdays(cohort)

        skill_info = get_statistical_info_about_iterable(skill_amounts, "skill_quantity")
        hours_info = get_statistical_info_about_iterable(hours_amounts, "hours")
        choice_info = get_statistical_info_about_iterable(choices_amounts, "choices")
        hamming_info = get_statistical_info_about_iterable(hamming_distances, "hamming")
        weekdays_info = get_statistical_info_about_iterable(weekdays, "weekdays")

        result = skill_info | hours_info | choice_info | hamming_info | weekdays_info
        result["shared_skill_amount"] = get_amount_of_shared_skills(cohort)
        result["skill_diversity"] = get_skill_variation(cohort)
        result["control"] = control
        result["size"] = len(cohort)
        return result


    def get_rows_from_cohorts(self, cohorts: list[Cohort], is_control = False) -> list[dict]:
        return [self.get_row_for_cohort(cohort, is_control) for cohort in cohorts]


    def cohorts_into_csv(self, csv_file_name: str,
                         real_cohorts: list[Cohort],
                         control_cohorts: list[Cohort],
                         remove_trivial_ps = True,
                         verbose = False):
        if remove_trivial_ps:
            real_cohorts = [cohort for cohort in real_cohorts if len(cohort) > 1]
            control_cohorts = [cohort for cohort in control_cohorts if len(cohort) > 1]
        with announce("Converting the cohorts into rows", verbose):
            rows = self.get_rows_from_cohorts(real_cohorts, False)
            rows.extend(self.get_rows_from_cohorts(control_cohorts, True))

        headers = rows[0].keys()
        with open(csv_file_name, "w+", newline="") as file:
            utils.make_folder_if_not_present(csv_file_name)
            csv_writer = csv.DictWriter(file, fieldnames=headers)
            csv_writer.writeheader()
            csv_writer.writerows(rows)



def test_and_produce_patterns(benchmark_problem: BTProblem,
                              csv_file_name: str,
                              pRef_size: int,
                              method: Literal["uniform", "FSStochasticSearch", "SA"],
                              verbose = False):
    pRef = get_history_pRef(benchmark_problem=benchmark_problem,
                            which_algorithm = method,
                            sample_size = pRef_size,
                            verbose=verbose)

    if verbose:
        print("Here is the pRef, in case you were curious")
        plot_solutions_in_pRef(pRef)

    metrics = [Simplicity(), MeanFitness(), Atomicity()]
    with announce("Running the PS_mining algorithm", verbose = verbose):
        toolbox = get_toolbox_for_problem(benchmark_problem,
                                          metrics, uses_experimental_crowding=True,
                                          pRef = pRef)
        final_population, logbook = nsga(toolbox=toolbox,
                                         mu =200,
                                         cxpb=0.5,
                                         mutpb=1/benchmark_problem.search_space.amount_of_parameters,
                                         ngen=2,
                                         stats=get_stats_object(),
                                         verbose=verbose)
    if verbose:
        print("The last population is ")
        report_in_order_of_last_metric(final_population, benchmark_problem)


    detector = BTProblemPatternDetector(benchmark_problem)
    cohorts = [detector.ps_to_cohort(ps) for ps in final_population]
    with announce(f"Analysing data and writing to file {csv_file_name}", verbose):
        detector.cohorts_into_csv(csv_file_name, cohorts)

    if verbose:
        print("Either way, the cohorts were:>>")

    cohorts_as_json = [cohort_to_json(cohort) for cohort in cohorts]
    print(json.dumps(cohorts_as_json))


def mine_pss_from_problem(benchmark_problem: BTProblem,
                              method: Literal["uniform", "FSStochasticSearch", "SA"],
                              pRef_size: int,
                              nsga_pop_size: int,
                              nsga_ngens: int,
                              verbose=True) -> (list[PS], list[tuple], Logbook):
    pRef = get_history_pRef(benchmark_problem=benchmark_problem,
                            which_algorithm = method,
                            sample_size = pRef_size,
                            verbose=verbose)

    if verbose:
        plot_solutions_in_pRef(pRef)

    metrics = [Simplicity(), MeanFitness(), Atomicity()]  # this is not actually used...
    with announce("Running the PS_mining algorithm", verbose = verbose):
        toolbox = get_toolbox_for_problem(benchmark_problem,
                                          metrics,
                                          algorithm="NSGAIII",
                                          uses_experimental_crowding=True,
                                          pRef = pRef)

        final_population, logbook = nsga(toolbox=toolbox,
                                         mu =nsga_pop_size,
                                         cxpb=0.5,
                                         mutpb=1/benchmark_problem.search_space.amount_of_parameters,
                                         ngen=nsga_ngens,
                                         stats=get_stats_object(),
                                         verbose=verbose)

    scores = [utils.as_float_tuple(ps.fitness.values) for ps in final_population]

    return final_population, scores, logbook

def mine_cohorts_from_problem(benchmark_problem: BTProblem,
                              method: Literal["uniform", "FSStochasticSearch", "SA"],
                              pRef_size: int,
                              nsga_pop_size: int,
                              nsga_ngens: int,
                              verbose=True) -> (list[Cohort], list[tuple], Logbook):
    pRef = get_history_pRef(benchmark_problem=benchmark_problem,
                            which_algorithm = method,
                            sample_size = pRef_size,
                            verbose=verbose)

    if verbose:
        plot_solutions_in_pRef(pRef)

    metrics = [Simplicity(), MeanFitness(), Atomicity()]  # this is not actually used...
    with announce("Running the PS_mining algorithm", verbose = verbose):
        toolbox = get_toolbox_for_problem(benchmark_problem,
                                          metrics,
                                          algorithm="NSGAIII",
                                          uses_experimental_crowding=True,
                                          pRef = pRef)

        final_population, logbook = nsga(toolbox=toolbox,
                                         mu =nsga_pop_size,
                                         cxpb=0.5,
                                         mutpb=1/benchmark_problem.search_space.amount_of_parameters,
                                         ngen=nsga_ngens,
                                         stats=get_stats_object(),
                                         verbose=verbose)

    detector = BTProblemPatternDetector(benchmark_problem)
    cohorts = [detector.ps_to_cohort(ps) for ps in final_population]
    scores = [utils.as_float_tuple(ps.fitness.values) for ps in final_population]

    return cohorts, scores, logbook




def json_to_cohorts(json_file: str) -> list[Cohort]:
    with open(json_file, "r") as file:
        data = json.load(file)
    return [[CohortMember.from_json(element) for element in cohort]
            for cohort in data]


def cohorts_to_json(cohorts: list[Cohort]) -> JSON:
    return [cohort_to_json(cohort) for cohort in cohorts]



def analyse_data_from_json_cohorts(
        problem: BTProblem,
        real_cohorts: list[Cohort],
        control_cohorts: list[Cohort],
        output_csv_file_name: str,
        verbose = True):
    detector = BTProblemPatternDetector(problem)
    with announce(f"Writing the data about the {len(real_cohorts)}+{len(control_cohorts)} into the file {output_csv_file_name}", verbose):
        detector.cohorts_into_csv(real_cohorts=real_cohorts,
                                  control_cohorts=control_cohorts,
                                  csv_file_name=output_csv_file_name,
                                  remove_trivial_ps=True)


def generate_control_data_for_cohorts(problem: BTProblem,
                                      reference_cohorts: list[Cohort],
                                      output_csv_file_name: str):
    detector = BTProblemPatternDetector(problem)
    control_cohorts = detector.generate_matching_random_cohorts(reference_cohorts,
                                                                amount_to_generate=10 * len(reference_cohorts))

    with announce(f"Writing the data about the REFERENCE COHORTS {len(reference_cohorts)} into the file {output_csv_file_name}"):
        detector.cohorts_into_csv(cohorts=control_cohorts, csv_file_name=output_csv_file_name, is_control=True)

    print("All finished")


def generate_coverage_stats(problem: BTProblem,
                            cohorts: list[Cohort]) -> dict:
    result = {worker.worker_id: 0 for worker in problem.workers}
    def register_cohort(cohort: Cohort):
        for member in cohort:
            worker_id = member.worker.worker_id
            result[worker_id] = result[worker_id] + 1

    for cohort in cohorts:
        register_cohort(cohort)
    return result



def get_shap_values_plot(csv_file: str, image_file_destination: str):
    print(f"Obtaining SHAP values for file {csv_file}")
    data = pd.read_csv(csv_file)

    # Define the features and target variable
    target_column = "control"  # Change to your target column
    features = data.columns.drop(target_column)

    X = data[features]
    y = data[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

    # Initialize and train the Random Forest model
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Initialize SHAP explainer and compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Plot the SHAP summary plot
    shap.summary_plot(shap_values, X)

    # Save the SHAP summary plot as an image
    plt.savefig(image_file_destination)
    print(f"SHAP summary plot saved as {image_file_destination}")

    # Optionally, display the image (if running in Jupyter or similar environments)
    plt.show()
