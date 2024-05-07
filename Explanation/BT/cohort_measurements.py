import itertools

import numpy as np

from Explanation.BT.Cohort import Cohort
from Explanation.BT.CohortMember import CohortMember


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
    total_pattern: np.ndarray = np.array(sum(member.chosen_rota_extended for member in cohort))
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