import itertools
import os
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.stats import t

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core import TerminationCriteria
from Core.PRef import PRef
from Core.PS import PS
from Explanation.detection_utils import generate_control_PSs
from PSMiners.AbstractPSMiner import AbstractPSMiner
from PSMiners.Mining import get_history_pRef, get_ps_miner, write_evaluated_pss_to_file, load_pss, write_pss_to_file
from utils import announce


class Detector:
    problem: BenchmarkProblem

    pRef_file: str   # npz
    ps_file: str     # npz
    control_ps_file: str   # npz
    properties_file: str   # csv
    verbose: bool


    cached_pRef: Optional[PRef]
    cached_pRef_mean: Optional[float]
    cached_pss: Optional[list[PS]]
    cached_control_pss: Optional[list[PS]]
    cached_properties: Optional[pd.DataFrame]

    speciality_threshold: float

    def __init__(self,
                 problem: BenchmarkProblem,
                 pRef_file: str,
                 ps_file: str,
                 control_ps_file: str,
                 properties_file: str,
                 speciality_threshold: float,
                 verbose = False):
        self.problem = problem
        self.pRef_file = pRef_file
        self.ps_file = ps_file
        self.control_ps_file = control_ps_file
        self.properties_file = properties_file
        self.speciality_threshold = speciality_threshold
        self.verbose = verbose

        self.cached_pRef = None
        self.cached_pRef_mean = None
        self.cached_pss = None
        self.cached_control_pss = None
        self.cached_properties = None


    @classmethod
    def with_folder(cls,
                    problem: BenchmarkProblem,
                    folder: str,
                    speciality_threshold = 0.1,
                    verbose = False):
        pRef_file = os.path.join("pRef.npz")
        ps_file = os.path.join("mined_ps.npz")
        control_ps_file = os.path.join("control_ps.npz")
        properties_file = os.path.join("ps_properties.csv")

        return cls(problem = problem,
                   pRef_file = pRef_file,
                   ps_file = ps_file,
                   control_ps_file = control_ps_file,
                   properties_file = properties_file,
                   speciality_threshold = speciality_threshold,
                   verbose=verbose)

    def generate_pRef(self,
                      sample_size: int,
                      which_algorithm: Literal["uniform", "GA", "SA", "GA_best", "SA_best"]):
        pRef  = get_history_pRef(benchmark_problem=self.problem,
                                 which_algorithm=which_algorithm,
                                 sample_size=sample_size,
                                 verbose=self.verbose)
        pRef.save(file=self.pRef_file)

        self.cached_pRef = pRef
        self.cached_pRef_mean = np.average(pRef.fitness_array)

    @property
    def pRef(self) -> PRef:
        if self.cached_pRef is None:
            with announce(f"Loading the cached pRef from {self.pRef_file}"):
                self.cached_pRef = PRef.load(self.pRef_file)
                self.cached_pRef_mean = np.average(self.cached_pRef.fitness_array)
        return self.cached_pRef

    @property
    def pRef_mean(self) -> float:
        if self.cached_pRef_mean is None:
            self.cached_pRef_mean = np.average(self.pRef.fitness_array)
        return self.cached_pRef_mean


    def generate_pss(self,
                     ps_miner_method : Literal["classic", "NSGA", "NSGA_experimental_crowding"] = "NSGA_experimental_crowding",
                     ps_budget: int = 10000):
        algorithm = get_ps_miner(self.pRef, which=ps_miner_method)

        with announce(f"Running {algorithm} on {self.pRef} with {ps_budget =}", self.verbose):
            termination_criterion = TerminationCriteria.PSEvaluationLimit(ps_limit=ps_budget)
            algorithm.run(termination_criterion, verbose=self.verbose)

        result_ps = algorithm.get_results(None)
        result_ps = AbstractPSMiner.without_duplicates(result_ps)

        with announce(f"Writing the PSs onto {self.ps_file}"):
            write_evaluated_pss_to_file(result_ps, self.ps_file)

        self.cached_pss = result_ps

    @property
    def pss(self) -> list[PS]:
        if self.cached_pss is None:
            with announce(f"Loading the cached pss from {self.ps_file}"):
                self.cached_pss = load_pss(self.ps_file)
        return self.cached_pss

    def generate_control_pss(self):
        control_pss = generate_control_PSs(self.problem.search_space, reference_pss=self.pss)
        write_pss_to_file(control_pss, self.control_ps_file)
        self.cached_control_pss = control_pss


    @property
    def control_pss(self) -> list[PS]:
        if self.cached_control_pss is None:
            with announce(f"Loading the control pss from {self.ps_file}", self.verbose):
                self.cached_control_pss = load_pss(self.control_ps_file)
        return self.cached_control_pss


    def ps_to_properties(self, ps: PS) -> dict:
        raise NotImplemented(f"An implementation of Detector does not implement .ps_to_properties")

    def make_properties_csv_file(self):
        properties_dicts = [self.ps_to_properties(ps) for ps in itertools.chain(self.pss, self.control_pss)]
        properties_df = pd.DataFrame(properties_dicts)
        properties_df["control"] = np.array([index < len(self.pss) for index in range(len(properties_dicts))])   # not my best work

        properties_df.to_csv(self.properties_file)
        self.cached_properties = properties_df

    @property
    def properties(self):
        if self.cached_properties is None:
            with announce(f"Loading the properties from {self.properties_file}", self.verbose):
                self.cached_properties = pd.read_csv(self.properties_file)
        return self.cached_properties


    def relative_property_ranking_within_dataset(self, property_name: str, property_value) -> float:
        properties_df = self.properties
        properties_df.sort_values(by=property_value)
        index = properties_df[property_name].searchsorted(property_value)
        return index / properties_df.shape[0]




    def relative_property_rank_is_significant(self, position: float) -> Optional[str]:
        is_low = position < self.speciality_threshold
        is_high = position > (1-self.speciality_threshold)

        if is_low:
            return "less"
        elif is_high:
            return "more"
        else:
            return None


    def t_test_for_mean_with_ps(self, ps: PS) -> (float, float):
        observations = self.pRef.fitnesses_of_observations(ps)
        n = len(observations)
        sample_mean = np.average(observations)
        sample_stdev = np.std(observations)

        if n < 1 or sample_stdev == 0:
            return -1, -1

        t_score = (sample_mean - self.cached_pRef_mean) / (sample_stdev / np.sqrt(n))
        p_value = 1 - t.cdf(abs(t_score), df=n - 1)
        return p_value, sample_mean

    def get_average_when_present_and_absent(self, ps: PS) -> (float, float):
        p_value, _ = self.t_test_for_mean_with_ps(ps)
        observations, not_observations = self.pRef.fitnesses_of_observations_and_complement(ps)
        return np.average(observations), np.average(not_observations)


    def only_significant_properties(self, all_properties: dict) -> list[(str, float, float)]:
        # obtain the relative_property_rankings
        items = [(key, value, self.relative_property_ranking_within_dataset(key, value))
                 for key, value in all_properties.items()]

        # only keep the ones that are significant
        items = [(key, value, relative_property_rank) for (key, value, relative_property_rank) in items
                 if self.relative_property_rank_is_significant(relative_property_rank)]
        return items

    def get_ps_description(self, ps: PS, ps_properties: dict) -> str:
        p_value, _ = self.t_test_for_mean_with_ps(ps)
        avg_when_present, avg_when_absent = self.get_average_when_present_and_absent(ps)
        significant_properties = self.only_significant_properties(ps_properties)


        avg_with_and_without_str =  (f"avg when present = {avg_when_present:.2f}, "
                                     f"avg when absent = {avg_when_absent:.2f}, "
                                     f"p-value = {p_value:e}")

        def repr_property(kvr) -> str:
            key, value, rank = kvr
            if rank > 0.5:
                return f"{rank} = {value:.2f} is relatively high (top {int(rank*100)}%)"
            else:
                return f"{rank} = {value:.2f} is relatively low (bottom {int((1-rank)*100)}%)"


        properties_str = "\n".join(repr_property(kvr) for kvr in significant_properties)

        return utils.indent(avg_with_and_without_str + "\n"+properties_str)



