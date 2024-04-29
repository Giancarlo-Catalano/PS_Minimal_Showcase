
"""
This file is uniquely to implement the function get_S_MF_A,
which stands for get Simplicity, Mean Fitness, Atomicity

In simple terms, there is a lot of redundancy in calculating the various observations for a ps for these 3 metrics,
and by calculating the PRefs together we can save a lot of time.
"""
from typing import TypeAlias, Optional

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PRef import PRef
from PS import PS, STAR
from PSMetric.Atomicity import Atomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from custom_types import ArrayOfFloats
from utils import announce



class RowsOfPRef:
    fsm: np.ndarray
    fitnesses: Optional[ArrayOfFloats]
    normalised_fitnesses: ArrayOfFloats


    def __init__(self, fsm: np.ndarray, fitnesses: ArrayOfFloats, normalised_fitnesses: ArrayOfFloats):
        self.fsm = fsm
        self.fitnesses = fitnesses
        self.normalised_fitnesses = normalised_fitnesses

    @classmethod
    def all_from_pRef(cls, pRef: PRef, normalised_fitnesses: ArrayOfFloats):
        fsm = pRef.full_solution_matrix.copy()
        fitnesses = pRef.fitness_array.copy()
        normalised_fitnesses = normalised_fitnesses.copy()
        return cls(fsm, fitnesses, normalised_fitnesses)


    def invalidate_fitnesses(self):
        self.fitnesses = None


    def filter_by_var_val(self, var: int, val: int):
        which = self.fsm[:, var] == val
        self.fsm = self.fsm[which]
        if self.fitnesses is not None:
            self.fitnesses = self.fitnesses[which]
        self.normalised_fitnesses = self.normalised_fitnesses[which]


    def get_mean_fitness(self) -> float:
        if self.fitnesses is None:
            raise ValueError("in RowsOfPRef, fitnesses is None")

        if len(self.fitnesses) == 0:
            return -np.inf
        return np.average(self.fitnesses)

    def get_normalised_mean_fitness(self) -> float:
        return float(np.sum(self.normalised_fitnesses))

    def copy(self):
        return RowsOfPRef(self.fsm, self.fitnesses, self.normalised_fitnesses)






class Classic3PSMetrics:
    pRef: PRef
    normalised_fitnesses: ArrayOfFloats
    cached_isolated_benefits: list[list[float]]

    def __init__(self, pRef: PRef):
        self.pRef = pRef
        self.normalised_fitnesses = self.get_normalised_fitness_array(self.pRef.fitness_array)
        self.cached_isolated_benefits = self.calculate_isolated_benefits()

    @classmethod
    def get_normalised_fitness_array(cls, fitness_array: ArrayOfFloats) -> ArrayOfFloats:
        min_fitness = np.min(fitness_array)
        normalised_fitnesses = fitness_array - min_fitness
        sum_fitness = np.sum(normalised_fitnesses, dtype=float)

        if sum_fitness == 0:
            raise Exception(f"The sum of fitnesses is 0, could not normalise")

        normalised_fitnesses /= sum_fitness
        return normalised_fitnesses


    def mf_of_rows(self, which_rows: RowsOfPRef)->float:
        return which_rows.get_mean_fitness()

    def normalised_mf_of_rows(self, which_rows: RowsOfPRef) -> float:
        return which_rows.get_normalised_mean_fitness()

    def calculate_isolated_benefits(self) -> list[list[float]]:
        """Requires self.normalised_pRef"""
        def benefit_when_isolating(var: int, val: int) -> float:
            relevant_rows = self.pRef.full_solution_matrix[:, var] == val
            return float(np.sum(self.normalised_fitnesses[relevant_rows]))

        ss = self.pRef.search_space
        return [[benefit_when_isolating(var, val)
                 for val in range(ss.cardinalities[var])]
                for var in range(ss.amount_of_parameters)]

    def get_simplicity_of_PS(self, ps: PS) -> float:
        return float(np.sum(ps.values == STAR))

    def get_relevant_rows_for_ps(self, ps: PS) -> (RowsOfPRef, list[RowsOfPRef]):
        """Returns the mean rows for ps, and the rows for the simplifications of ps"""

        """Eg for * 1 2 3 *, it returns
           rows(* 1 2 3*), [rows(* * 2 3), rows(* 1 * 3 *), rows(* 1 2 * *)]
        """

        fsm = self.pRef.full_solution_matrix

        def subset_where_column_has_value(superset: RowsOfPRef, variable: int, value: int) -> RowsOfPRef:
            result = superset.copy()
            result.filter_by_var_val(variable, value)
            return result


        with_all_fixed = RowsOfPRef.all_from_pRef(self.pRef, normalised_fitnesses=self.normalised_fitnesses)
        except_one_fixed = []

        for var in ps.get_fixed_variable_positions():
            value = ps[var]
            except_one_fixed = [subset_where_column_has_value(original, var, value)
                      for original in except_one_fixed]
            except_one_fixed.append(with_all_fixed.copy())
            with_all_fixed = subset_where_column_has_value(with_all_fixed, var, value)

        return with_all_fixed, except_one_fixed

    def get_relevant_isolated_benefits(self, ps: PS) -> ArrayOfFloats:
        return np.array([self.cached_isolated_benefits[var][val]
                         for var, val in enumerate(ps.values)
                         if val != STAR])

    def get_atomicity_from_relevant_rows(self, ps: PS,
                                         rows_of_all_fixed: RowsOfPRef,
                                         except_for_one: list[RowsOfPRef]) -> float:
        pAB = self.normalised_mf_of_rows(rows_of_all_fixed)
        if pAB == 0.0:
            return pAB

        isolated = self.get_relevant_isolated_benefits(ps)
        excluded = np.array([self.normalised_mf_of_rows(rows) for rows in except_for_one])

        if len(isolated) == 0:  # ie we have the empty ps
            return 0

        max_denominator = np.max(isolated * excluded)  # praying that they are always the same size!

        result = pAB * np.log(pAB / max_denominator)
        if np.isnan(result).any():
            raise Exception("There is a nan value returned in atomicity")
        return result


    def get_S_MF_A(self, ps: PS) -> (float, float, float):
        rows_all_fixed, excluding_one = self.get_relevant_rows_for_ps(ps)

        simplicity = self.get_simplicity_of_PS(ps)
        mean_fitness = self.mf_of_rows(rows_all_fixed)
        atomicity = self.get_atomicity_from_relevant_rows(ps,
                                                          rows_all_fixed,
                                                          excluding_one)
        return simplicity, mean_fitness, atomicity




def test_classic3(benchmark_problem: BenchmarkProblem,sample_size: int):
    pRef = benchmark_problem.get_reference_population(sample_size)

    metrics = [Simplicity(), MeanFitness(), Atomicity()]
    for metric in metrics:
        metric.set_pRef(pRef)
    classic3 = Classic3PSMetrics(pRef)


    def get_control_values(ps: PS) -> (float, float, float):
        return np.array(tuple(metric.get_single_score(ps) for metric in metrics))

    def get_experimental_value(ps: PS) -> (float, float, float):
        return np.array(classic3.get_S_MF_A(ps))


    test_quantity = 6
    for _ in range(test_quantity):
        ps = PS.random(benchmark_problem.search_space, True)
        control = get_control_values(ps)
        experimental = get_experimental_value(ps)

        error = control-experimental
        print(f"The PS is {ps}, error = {error}") # , control = {control}, experimental = {experimental}")


    pss_to_evaluate = [PS.random(benchmark_problem.search_space, half_chance_star=True)
                       for _ in range(10000)]

    with announce(f"Calculating using the traditional metrics"):
        control_results = [get_control_values(ps) for ps in pss_to_evaluate]

    with announce(f"Calculating using the new metrics"):
        experimental_results = [get_experimental_value(ps) for ps in pss_to_evaluate]

    def significant_difference(cont, exper) -> bool:
        error = abs(cont - exper)
        if not all(np.isreal(error)):
            return True
        return sum(error) > 0.000001


    for ps, c, e in zip(pss_to_evaluate, control_results, experimental_results):
        if significant_difference(c, e):
            print(f"The {ps} has a significant error: {c} vs {e}")






