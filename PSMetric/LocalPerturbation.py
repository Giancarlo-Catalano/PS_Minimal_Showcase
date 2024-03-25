import itertools
import warnings
from typing import Optional

import numpy as np

from PRef import PRef
from PS import PS, STAR
from PSMetric.Metric import Metric
from custom_types import ArrayOfBools, ArrayOfFloats


class LocalPerturbationCalculator:
    pRef: PRef
    cached_value_locations: list[list[ArrayOfBools]]

    def __init__(self, pRef: PRef):
        self.pRef = pRef
        self.cached_value_locations = self.get_cached_value_locations(pRef)

    @staticmethod
    def get_cached_value_locations(pRef: PRef) -> list[list[ArrayOfBools]]:
        def get_where_var_val(var: int, val: int) -> ArrayOfBools:
            return pRef.full_solution_matrix[:, var] == val

        return [[get_where_var_val(var, val)
                 for val in range(cardinality)]
                for var, cardinality in enumerate(pRef.search_space.cardinalities)]

    def get_univariate_perturbation_fitnesses(self, ps: PS, locus: int) -> (ArrayOfFloats, ArrayOfFloats):
        """ The name is horrible, but essentially it returns
           (fitnesses of observations of ps,  fitnesses of observations which match ps but at locus it DOESN't match"""

        assert (ps.values[locus] != STAR)

        where_ps_matches_ignoring_locus = np.full(shape=self.pRef.sample_size, fill_value=True, dtype=bool)
        for var, val in enumerate(ps.values):
            if val != STAR and var != locus:
                where_ps_matches_ignoring_locus = np.logical_and(where_ps_matches_ignoring_locus,
                                                                 self.cached_value_locations[var][val])

        locus_val = ps.values[locus]

        where_locus = self.cached_value_locations[locus][locus_val]
        where_value_matches = np.logical_and(where_ps_matches_ignoring_locus, where_locus)
        where_complement_matches = np.logical_and(where_value_matches, np.logical_not(where_locus))

        return (self.pRef.fitness_array[where_value_matches], self.pRef.fitness_array[where_complement_matches])

    def get_bivariate_perturbation_fitnesses(self, ps: PS, locus_a: int, locus_b) -> (ArrayOfFloats, ArrayOfFloats):
        """ returns the fitnesses of x(a, b), x(not a, b), x(a, not b), x(not a, not b)"""

        assert (ps.values[locus_a] != STAR)
        assert (ps.values[locus_b] != STAR)

        where_ps_matches_ignoring_loci = np.full(shape=self.pRef.sample_size, fill_value=True, dtype=bool)
        for var, val in enumerate(ps.values):
            if val != STAR and var != locus_a and var != locus_b:
                where_ps_matches_ignoring_loci = np.logical_and(where_ps_matches_ignoring_loci,
                                                                self.cached_value_locations[var][val])

        val_a = ps.values[locus_a]
        val_b = ps.values[locus_b]

        where_a = self.cached_value_locations[locus_a][val_a]
        where_b = self.cached_value_locations[locus_b][val_b]
        where_not_a = np.logical_not(where_a)
        where_not_b = np.logical_not(where_b)

        where_a_b = np.logical_and(where_a, where_b)
        where_not_a_b = np.logical_and(where_not_a, where_b)
        where_a_not_b = np.logical_and(where_a, where_not_b)
        where_not_a_not_b = np.logical_and(where_not_a, where_not_b)

        def fits(where_condition: ArrayOfBools):
            return self.pRef.fitness_array[np.logical_and(where_ps_matches_ignoring_loci, where_condition)]

        return fits(where_a_b), fits(where_not_a_b), fits(where_a_not_b), fits(where_not_a_not_b)

    def get_delta_f_of_ps_at_locus_univariate(self, ps: PS, locus: int) -> float:
        value_matches, complement_matches = self.get_univariate_perturbation_fitnesses(ps, locus)

        if len(value_matches) == 0 or len(complement_matches) == 0:
            warnings.warn(
                f"Encountered a PS with insufficient observations when calculating Univariate Local perturbation")
            return 0  # panic

        return np.average(value_matches) - np.average(complement_matches)

    def get_delta_f_of_ps_at_loci_bivariate(self, ps: PS, locus_a: int, locus_b: int) -> float:
        fs = self.get_bivariate_perturbation_fitnesses(ps, locus_a, locus_b)
        fs_yy, fs_ny, fs_yn, fs_nn = fs
        if any(len(fs) == 0 for fs in fs):
            warnings.warn(
                f"Encountered a PS with insufficient observations ({ps}) when calculating bivariate Local perturbation")
            return 0  # panic

        f_yy = np.average(fs_yy)
        f_yn = np.average(fs_yn)
        f_ny = np.average(fs_ny)
        f_nn = np.average(fs_nn)

        return abs(f_yy + f_nn - f_yn - f_ny)


class UnivariateLocalPerturbation(Metric):
    linkage_calculator: Optional[LocalPerturbationCalculator]
    def __init__(self):
        self.pRef = None
        super().__init__()

    def __repr__(self):
        return "UnivariateLocalPerturbation"

    def set_pRef(self, pRef: PRef):
        self.linkage_calculator = LocalPerturbationCalculator(pRef)

    def get_single_score(self, ps: PS) -> float:
        fixed_loci = ps.get_fixed_variable_positions()
        dfs = [self.linkage_calculator.get_delta_f_of_ps_at_locus_univariate(ps, i) for i in fixed_loci]
        return min(dfs)


class BivariateLocalPerturbation(Metric):
    linkage_calculator: Optional[LocalPerturbationCalculator]

    def __init__(self):
        self.pRef = None
        super().__init__()

    def __repr__(self):
        return "BivariateLocalPerturbation"

    def set_pRef(self, pRef: PRef):
        self.linkage_calculator = LocalPerturbationCalculator(pRef)

    def get_single_score(self, ps: PS) -> float:
        fixed_loci = ps.get_fixed_variable_positions()
        pairs = list(itertools.combinations(fixed_loci, r=2))
        dfs = [self.linkage_calculator.get_delta_f_of_ps_at_loci_bivariate(ps, a, b) for a, b in pairs]
        return min(dfs)




