import itertools
from typing import TypeAlias, Optional

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison

import utils
from PRef import PRef
from PS import PS, STAR
from PSMetric.Metric import Metric
from bioinfokit.analys import stat

LinkageTable: TypeAlias = np.ndarray


class SecondLinkage(Metric):
    linkage_table: Optional[LinkageTable]
    normalised_linkage_table: Optional[LinkageTable]

    def __init__(self):
        super().__init__()
        self.linkage_table = None
        self.normalised_linkage_table = None

    def __repr__(self):
        return "SecondLinkage"

    def set_pRef(self, pRef: PRef):
        self.linkage_table = self.get_linkage_table(pRef)
        self.normalised_linkage_table = self.get_normalised_linkage_table(self.linkage_table)

    @staticmethod
    def get_linkage_table(pRef: PRef) -> LinkageTable:

        """ this is crazy slow."""
        var_count = pRef.search_space.amount_of_parameters
        dependent_var = "fitness"
        independent_vars = [f"var_{index}" for index in range(var_count)]

        dataframe = pd.DataFrame(
            [row.tolist() + [fitness] for row, fitness in zip(pRef.full_solution_matrix, pRef.fitness_array)])
        dataframe.rename(columns={index: f"var_{index}" for index in range(var_count)}, inplace=True)
        dataframe.rename(columns={var_count: dependent_var}, inplace=True)

        variable_pairs = list(itertools.combinations(independent_vars, 2))

        results = []
        for pair in variable_pairs:
            # Reshape the data into long format for the pair of variables
            df_long = dataframe[[dependent_var, *pair]].melt(id_vars=[dependent_var], value_vars=pair, var_name='variable',
                                                      value_name='value')

            # Fit ANOVA model
            formula = f"{dependent_var} ~ C(variable) + C(value)"
            model = ols(formula, data=df_long).fit()

            # Perform Tukey's HSD test
            mc = MultiComparison(df_long[dependent_var], df_long['value'])
            tukey_result = mc.tukeyhsd()

            # Store the results
            results.append({'Pair': pair, 'ANOVA': model, 'Tukey_HSD': tukey_result})

        # TODO
        linkage_table = None

        # then we mirror it for convenience...
        upper_triangle = np.triu(linkage_table, k=1)
        linkage_table = linkage_table + upper_triangle.T
        return linkage_table

    @staticmethod
    def get_normalised_linkage_table(linkage_table: LinkageTable):
        where_to_consider = np.triu(np.full_like(linkage_table, True, dtype=bool), k=1)
        triu_min = np.min(linkage_table, where=where_to_consider, initial=np.inf)
        triu_max = np.max(linkage_table, where=where_to_consider, initial=-np.inf)
        normalised_linkage_table: LinkageTable = (linkage_table - triu_min) / triu_max

        return normalised_linkage_table

    def get_linkage_scores(self, ps: PS) -> np.ndarray:
        fixed = ps.values != STAR
        fixed_combinations: np.array = np.outer(fixed, fixed)
        fixed_combinations = np.triu(fixed_combinations, k=1)
        return self.linkage_table[fixed_combinations]

    def get_normalised_linkage_scores(self, ps: PS) -> np.ndarray:
        fixed = ps.values != STAR
        fixed_combinations: np.array = np.outer(fixed, fixed)
        fixed_combinations = np.triu(fixed_combinations, k=0)
        return self.normalised_linkage_table[fixed_combinations]

    def get_single_score_using_avg(self, ps: PS) -> float:
        if ps.fixed_count() < 1:
            return 0
        else:
            return utils.harmonic_mean(self.get_linkage_scores(ps))

    def get_single_normalised_score(self, ps: PS) -> float:
        if ps.fixed_count() < 1:
            return 0
        else:
            return np.average(self.get_normalised_linkage_scores(ps))

    def get_single_score(self, ps: PS) -> float:
        return self.get_single_score_using_avg(ps)
