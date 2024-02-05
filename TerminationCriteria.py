from enum import Enum, auto
from typing import Iterable, Any

from custom_types import Fitness


class TerminationCriteria:
    def __init__(self):
        pass


    def __repr__(self):
        raise Exception("Implementation of TerminationCriteria does not implement __repr__")

    def termination_condition_met(self, **kwargs):
        raise Exception("Implementation of TerminationCriteria does not implement termination_criteria_met")

class EvaluationBudgetLimit(TerminationCriteria):
    max_evaluations: int
    def __init__(self, max_evaluations: int):
        super().__init__()
        self.max_evaluations = max_evaluations


    def __repr__(self):
        return f"EvaluationBudget({self.max_evaluations})"

    def termination_condition_met(self, **kwargs):
        return kwargs["evaluations"] >= self.max_evaluations



class TimeLimit(TerminationCriteria):
    max_time: float

    def __init__(self, max_time: float):
        super().__init__()
        self.max_time = max_time

    def __repr__(self):
        return f"TimeLimit({self.max_time})"

    def termination_condition_met(self, **kwargs):
        return kwargs["time"] >= self.max_time



class IterationLimit(TerminationCriteria):
    max_iterations: float

    def __init__(self, max_iterations: float):
        super().__init__()
        self.max_iterations = max_iterations

    def __repr__(self):
        return f"TimeLimit({self.max_iterations})"

    def termination_condition_met(self, **kwargs):
        return kwargs["iterations"] >= self.max_iterations




class UntilAllTargetsFound(TerminationCriteria):
    targets: Iterable[Any]

    def __init__(self, targets: Iterable[Any]):
        super().__init__()
        self.targets = targets

    def __repr__(self):
        return f"Targets({self.targets})"

    def termination_condition_met(self, **kwargs):
        would_be_returned = kwargs["return"]

        return all(target in would_be_returned for target in self.targets)



class UntilGlobalOptimaReached(TerminationCriteria):
    global_optima_fitness: Fitness

    def __init__(self, global_optima_fitness: Fitness):
        super().__init__()
        self.global_optima_fitness = global_optima_fitness

    def __repr__(self):
        return f"UntilGlobalOptima({self.global_optima_fitness})"

    def termination_condition_met(self, **kwargs):
        would_be_returned = kwargs["returned_fitnesses"]

        return self.global_optima_fitness in would_be_returned



class UnionOfCriteria(TerminationCriteria):
    subcriteria: list[TerminationCriteria]
    def __init__(self, *subcriteria):
        self.subcriteria = list(subcriteria)
        super().__init__()


    def __repr__(self):
        return "Union("+", ".join(f"{sc}" for sc in self.subcriteria)+")"


    def termination_condition_met(self, **kwargs):
        return any(sc.termination_criteria_met(**kwargs) for sc in self.subcriteria)

