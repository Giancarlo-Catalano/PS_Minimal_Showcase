from typing import TypeAlias

from Core.custom_types import JSON
from Explanation.BT.CohortMember import CohortMember

Cohort: TypeAlias = list[CohortMember]

def cohort_to_json(cohort: Cohort) -> JSON:
    return [member.to_json() for member in cohort]