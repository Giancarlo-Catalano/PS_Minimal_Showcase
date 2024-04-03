from EvaluatedFS import EvaluatedFS
from PRef import PRef
from PS import PS


def just_empty(pRef: PRef, quantity = None) -> list[PS]:
    return [PS.empty(pRef.search_space)]


def from_best_of_PRef(pRef: PRef, quantity: int) -> list[PS]:

    evaluated_fss: list[EvaluatedFS] = pRef.get_evaluated_FSs()
    evaluated_fss.sort(reverse=True)
    return [PS.from_FS(individual.full_solution)
            for individual in evaluated_fss[:quantity]]


def from_random(pRef: PRef, quantity: int) -> list[PS]:
    return [PS.random(pRef.search_space) for _ in range(quantity)]