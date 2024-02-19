import itertools
from typing import Optional

import utils
from BenchmarkProblems.BT.RotaPattern import RotaPattern
from BenchmarkProblems.BT.Worker import Worker, WorkerVariables
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FullSolution import FullSolution
from PS import PS, STAR
from SearchSpace import SearchSpace





class BTProblem(BenchmarkProblem):
    calendar_length: int
    workers: list[Worker]

    def __init__(self, workers: list[Worker], calendar_length: int):
        self.workers = workers
        self.calendar_length = calendar_length

        variable_cardinalities = itertools.chain(worker.get_variable_cardinalities() for worker in self.workers)
        super().__init__(SearchSpace(variable_cardinalities))


    def __repr__(self):
        return f"BTProblem(amount_of_workers = {self.workers}, calendar_length={self.calendar_length})"


    def get_variables_from_fs(self, fs: FullSolution) -> list[WorkerVariables]:
        broken_by_worker = utils.break_list(list(fs.values), 2)
        def list_to_wv(variable_list: list) -> WorkerVariables:
            which_rota, starting_day = variable_list
            return WorkerVariables(which_rota, starting_day)

        return [list_to_wv(list_vars) for list_vars in broken_by_worker]

    def get_variables_from_ps(self, ps: PS) -> list[WorkerVariables]:
        broken_by_worker = utils.break_list(list(ps.values), 2)

        def list_to_wv(variable_list: list) -> WorkerVariables:
            which_rota, starting_day = variable_list

            def from_value(value):
                if value == STAR:
                    return None
                else:
                    return value

            return WorkerVariables(from_value(which_rota), from_value(starting_day))

        return [list_to_wv(list_vars) for list_vars in broken_by_worker]



