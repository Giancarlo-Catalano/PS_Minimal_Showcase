import os

import numpy as np

from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.PS import PS
from Explanation.BT.Cohort import ps_to_cohort
from Explanation.BT.cohort_measurements import get_hamming_distances, get_ranges_in_weekdays
from Explanation.Detector import Detector


class BTDetector(Detector):
    problem: EfficientBTProblem

    def __init__(self,
                 problem: EfficientBTProblem,
                 folder: str,
                 speciality_threshold: float,
                 verbose = False):
        pRef_file = os.path.join(folder, "pRef.npz")
        ps_file = os.path.join(folder, "mined_ps.npz")
        control_ps_file = os.path.join(folder, "control_ps.npz")
        properties_file = os.path.join(folder, "ps_properties.csv")

        super(BTDetector, self).__init__(problem = problem,
                                       pRef_file = pRef_file,
                                       ps_file = ps_file,
                                       control_ps_file = control_ps_file,
                                       properties_file = properties_file,
                                       speciality_threshold = speciality_threshold,
                                       verbose=verbose)
    def ps_to_properties(self, ps: PS) -> dict:
        cohort = ps_to_cohort(self.problem, ps)

        mean_rota_choice_amount = np.average([member.get_amount_of_choices() for member in cohort])
        mean_amount_of_hours = np.average([member.get_amount_of_working_hours() for member in cohort])
        mean_hamming_distance = np.average(get_hamming_distances(cohort))
        local_fitness = np.average(get_ranges_in_weekdays(cohort))

        return {"mean_rota_choice_quantity": mean_rota_choice_amount,
                "mean_amount_of_hours": mean_amount_of_hours,
                "mean_difference_in_rotas": mean_hamming_distance,
                "local_fitness": local_fitness}



    def repr_property(self, property_name:str, property_value:str, property_rank_range:str):
        lower_rank, upper_rank = property_rank_range
        is_low = lower_rank < 0.5
        if property_name == "mean_rota_choice_quantity":
            return f"They have relatively {'few' if is_low else 'many'}' rota choices (mean = {property_value}"
        elif property_name == "mean_amount_of_hours":
            return f"The working hour amounts are very {'low' if is_low else 'high'}' (mean = {property_value}"
        elif property_name == "mean_difference_in_rotas":
            return f"The rotas are {'similar' if is_low else 'different'}' (rank = {int(property_rank_range[0]*100)}% ~ {int(property_rank_range[1]*100)}%)"
        elif property_name == "local_fitness":
            return f"The rotas {'' if is_low else 'do not'} complement each other (rank = {int(property_rank_range[0]*100)}% ~ {int(property_rank_range[1]*100)})%"