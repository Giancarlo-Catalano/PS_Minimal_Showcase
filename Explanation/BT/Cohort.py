from typing import TypeAlias

from BenchmarkProblems.BT.BTProblem import BTProblem
from Core.PS import PS, STAR
from Core.custom_types import JSON
from Explanation.BT.CohortMember import CohortMember

Cohort: TypeAlias = list[CohortMember]


def ps_to_cohort(problem: BTProblem, ps: PS) -> Cohort:
    def fixed_var_to_cohort_member(var: int, val: int) -> CohortMember:
        worker = problem.workers[var]
        return CohortMember(worker, val, calendar_length=problem.calendar_length)
    return [fixed_var_to_cohort_member(var, val)
            for var, val in enumerate(ps.values)
            if val != STAR]

def cohort_to_json(cohort: Cohort) -> JSON:
    return [member.to_json() for member in cohort]