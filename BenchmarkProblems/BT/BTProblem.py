import itertools
from typing import Optional, TypeAlias

import utils
from BenchmarkProblems.BT.ReadFromFiles import get_dicts_from_RPD, make_roster_patterns_from_RPD, get_dicts_from_ED, \
    make_employees_from_ED, get_skills_dict
from BenchmarkProblems.BT.RotaPattern import RotaPattern, get_workers_present_each_day_of_the_week, get_range_scores
from BenchmarkProblems.BT.Worker import Worker, WorkerVariables, Skill
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FullSolution import FullSolution
from PS import PS, STAR
from SearchSpace import SearchSpace
from resources.BT.names import names




class BTProblem(BenchmarkProblem):
    calendar_length: int
    workers: list[Worker]
    weights = [1, 1, 1, 1, 1, 10, 10]
    uses_custom_starting_days: bool
    can_switch_rotas: bool
    considers_skills: bool
    considers_areas: bool
    considers_times: bool
    all_skills: set[Skill]

    def __init__(self,
                 workers: list[Worker],
                 calendar_length: int,
                 custom_starting_days = False,
                 can_switch_rotas = False,
                 considers_skills = False,
                 considers_areas = False,
                 considers_times = False):

        self.uses_custom_starting_days = custom_starting_days
        self.can_switch_rotas = can_switch_rotas
        self.considers_skills = considers_skills
        self.considers_areas = considers_areas
        self.considers_times = considers_times
        self.workers = workers
        self.calendar_length = calendar_length

        assert(calendar_length % 7 == 0)

        variable_cardinalities = utils.join_lists(worker.get_variable_cardinalities(self.uses_custom_starting_days)
                                                  for worker in self.workers)
        super().__init__(SearchSpace(variable_cardinalities))

        self.all_skills = set(skill for worker in self.workers
                              for skill in worker.available_skills)

    @classmethod
    def from_csv_files(cls, employee_data_file: str, employee_skills_file: str, rota_file: str, calendar_length: int):
        rpd_dicts = get_dicts_from_RPD(rota_file)
        rotas = make_roster_patterns_from_RPD(rpd_dicts)

        employee_dicts = get_dicts_from_ED(employee_data_file)
        employee_skills = get_skills_dict(employee_skills_file)
        employees = make_employees_from_ED(employee_dicts, employee_skills, rotas, names)
        return cls(employees, calendar_length)

    @classmethod
    def from_default_files(cls):
        return cls.from_csv_files(employee_data_file=r"C:\Users\gac8\PycharmProjects\PS-PDF\resources\BT\employeeData.csv",
                                  employee_skills_file=r"C:\Users\gac8\PycharmProjects\PS-PDF\resources\BT\employeeSkillsData.csv",
                                  rota_file=r"C:\Users\gac8\PycharmProjects\PS-PDF\resources\BT\roster_pattern_days.csv",
                                  calendar_length=28)

    def to_json(self) -> dict:
        result = dict()
        result["calendar_length"] = self.calendar_length
        result["weights"] = self.weights
        result["custom_starting_days"] = self.uses_custom_starting_days
        result["can_switch_rotas"] = self.can_switch_rotas
        result["considers_skills"] = self.considers_skills
        result["considers_areas"] = self.considers_areas
        result["considers_times"] = self.considers_times
        result["workers"] = [worker.to_json() for worker in self.workers]
        return result

    def __repr__(self):
        return f"BTProblem(amount_of_workers = {len(self.workers)}, calendar_length={self.calendar_length})"

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
        rotas = [wv.effective_rota(worker, consider_starting_week=self.uses_custom_starting_days)
                 for wv, worker in zip(wvs, self.workers)]

        rotas_by_skill = {skill: [rota for rota, worker in zip(rotas, self.workers)
                                  if skill in worker.available_skills]
                          for skill in self.all_skills}

        def range_score_for_rotas(input_rotas) -> float:
            calendar = get_workers_present_each_day_of_the_week(input_rotas, self.calendar_length)
            ranges = get_range_scores(calendar)
            return sum(range_score*weight for range_score, weight in zip(ranges, self.weights))

        return sum(range_score_for_rotas(rotas_by_skill[skill]) for skill in rotas_by_skill)

    def fitness_function(self, fs: FullSolution) -> float:
        wvs = self.get_variables_from_fs(fs)
        return -self.get_range_score_from_wvs(wvs)  # the minus is there because this is a minimisation problem

    def get_amount_of_first_choices(self, wvs: list[WorkerVariables]) -> int:
        return len([wv for wv in wvs if wv.which_rota==0])