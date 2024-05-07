import os

import numpy as np

from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from Core.PS import PS, STAR
from Explanation.BT.Cohort import ps_to_cohort
from Explanation.BT.cohort_measurements import get_hamming_distances, get_ranges_in_weekdays
from Explanation.Detector import Detector


class BTDetector(Detector):
    problem: GraphColouring

    def __init__(self,
                 problem: GraphColouring,
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
        def contains_pair(pair):
            a, b = pair
            return ps[a] == STAR or ps[b] == STAR

        edge_count = len([pair for pair in self.problem.connections
                          if contains_pair(pair)])
        return {"edge_count": edge_count}

