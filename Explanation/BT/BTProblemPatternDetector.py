import csv
import csv
import itertools
import json
import random
from typing import TypeAlias, Iterable, Literal

import numpy as np
import pandas as pd
import shap
from deap.tools import Logbook
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import utils
from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.Worker import Worker
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import ExtendedPattern, \
    rota_to_extended_pattern
from Core.PRef import plot_solutions_in_pRef
from Core.PS import PS, STAR
from Core.PSMetric.Atomicity import Atomicity
from Core.PSMetric.MeanFitness import MeanFitness
from Core.PSMetric.Simplicity import Simplicity
from Core.custom_types import JSON
from Explanation.BT.Cohort import Cohort, cohort_to_json
from Explanation.BT.CohortMember import CohortMember
from Explanation.BT.cohort_measurements import get_hamming_distances, get_ranges_in_weekdays, \
    get_amount_of_shared_skills, get_skill_variation
from PSMiners.DEAP.deap_utils import get_toolbox_for_problem, nsga, get_stats_object, \
    report_in_order_of_last_metric
from PSMiners.Mining import get_history_pRef, load_pss
from utils import announce




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
        pss = load_pss(ps_file)
        cohorts = [self.ps_to_cohort(ps) for ps in pss]
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

    def get_row_for_cohort(self, cohort: Cohort, control: bool) -> dict:
        skill_amounts = [member.get_amount_of_skills() for member in cohort]
        hours_amounts = [member.get_amount_of_working_hours() for member in cohort]
        choices_amounts = [member.get_amount_of_choices() for member in cohort]
        hamming_distances = get_hamming_distances(cohort)
        weekdays = get_ranges_in_weekdays(cohort)

        skill_info = utils.get_statistical_info_about_iterable(skill_amounts, "skill_quantity")
        hours_info = utils.get_statistical_info_about_iterable(hours_amounts, "hours")
        choice_info = utils.get_statistical_info_about_iterable(choices_amounts, "choices")
        hamming_info = utils.get_statistical_info_about_iterable(hamming_distances, "hamming")
        weekdays_info = utils.get_statistical_info_about_iterable(weekdays, "weekdays")

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
