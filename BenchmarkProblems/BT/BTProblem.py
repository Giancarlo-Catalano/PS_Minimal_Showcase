import itertools
from typing import Optional

import utils
from BenchmarkProblems.BT.ReadFromFiles import get_dicts_from_RPD, make_roster_patterns_from_RPD, get_dicts_from_ED, \
    make_employees_from_ED
from BenchmarkProblems.BT.RotaPattern import RotaPattern, get_workers_present_each_day_of_the_week, get_range_scores
from BenchmarkProblems.BT.Worker import Worker, WorkerVariables
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FullSolution import FullSolution
from PS import PS, STAR
from SearchSpace import SearchSpace
from resources.BT.names import names


class BTProblem(BenchmarkProblem):
    calendar_length: int
    workers: list[Worker]
    weights = [1, 1, 1, 1, 1, 10, 10]
    custom_starting_days: bool
    can_switch_rotas: bool
    considers_skills: bool
    considers_areas: bool
    considers_times: bool

    def __init__(self,
                 workers: list[Worker],
                 calendar_length: int,
                 custom_starting_days = False,
                 can_switch_rotas = False,
                 considers_skills = False,
                 considers_areas = False,
                 considers_times = False):

        self.custom_starting_days = custom_starting_days
        self.can_switch_rotas = can_switch_rotas
        self.considers_skills = considers_skills
        self.considers_areas = considers_areas
        self.considers_times = considers_times
        self.workers = workers
        self.calendar_length = calendar_length

        assert(calendar_length % 7 == 0)

        variable_cardinalities = utils.join_lists(worker.get_variable_cardinalities(self.custom_starting_days)
                                                  for worker in self.workers)
        super().__init__(SearchSpace(variable_cardinalities))

    @classmethod
    def from_files(cls, employee_data_file: str, rota_file: str, calendar_length: int):
        rpd_dicts = get_dicts_from_RPD(rota_file)
        rotas = make_roster_patterns_from_RPD(rpd_dicts)

        employee_dicts = get_dicts_from_ED(employee_data_file)
        employees = make_employees_from_ED(employee_dicts, rotas, names)
        return cls(employees, calendar_length)

    def __repr__(self):
        return f"BTProblem(amount_of_workers = {self.workers}, calendar_length={self.calendar_length})"

    def get_variables_from_fs(self, fs: FullSolution) -> list[WorkerVariables]:
        broken_by_worker = utils.break_list(list(fs.values), 1)

        def list_to_wv(variable_list: list) -> WorkerVariables:
            which_rota = variable_list[0]
            return WorkerVariables(which_rota, starting_day=0)

        return [list_to_wv(list_vars) for list_vars in broken_by_worker]


    def get_variables_from_ps(self, ps: PS) -> list[WorkerVariables]:
        broken_by_worker = utils.break_list(list(ps.values), 1)

        def list_to_wv(variable_list: list) -> WorkerVariables:
            which_rota = variable_list[0]

            def from_value(value):
                if value == STAR:
                    return None
                else:
                    return value

            return WorkerVariables(from_value(which_rota), starting_day=0)

        return [list_to_wv(list_vars) for list_vars in broken_by_worker]

    def get_range_score_from_wvs(self, wvs: list[WorkerVariables]) -> float:
        """assumes that all of the wvs's are valid, ie that none of the attributes are None"""
        rotas = [wv.effective_rota(worker, consider_starting_week=self.custom_starting_days) for wv, worker in zip(wvs, self.workers)]
        calendar = get_workers_present_each_day_of_the_week(rotas, self.calendar_length)

        ranges = get_range_scores(calendar)
        return sum(range_score*weight for range_score, weight in zip(ranges, self.weights))

    def fitness_function(self, fs: FullSolution) -> float:
        wvs = self.get_variables_from_fs(fs)
        return -self.get_range_score_from_wvs(wvs)  # the minus is there because this is a minimisation problem

    def get_amount_of_first_choices(self, wvs: list[WorkerVariables]) -> int:
        return len([wv for wv in wvs if wv.which_rota==0])
