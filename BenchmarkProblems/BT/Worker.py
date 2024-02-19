from typing import Optional

import numpy as np

from BenchmarkProblems.BT.RotaPattern import RotaPattern


class Skill:
    skill_index: int

    def __init__(self, skill_index: int):
        self.skill_index = skill_index


class Worker:
    available_skills: set[Skill]
    available_rotas: list[RotaPattern]
    worker_id: int
    name: str

    def __init__(self,
                 available_skills: set[Skill],
                 available_rotas: list[RotaPattern],
                 worker_id: int,
                 name: str):
        self.available_skills = available_skills
        self.available_rotas = available_rotas
        self.worker_id = worker_id
        self.name = name

        assert(self.are_rotas_valid())

    def __repr__(self):
        return self.name


    def are_rotas_valid(self) -> bool:
        if len(self.available_rotas) < 1:
            return False

        rota_length = len(self.available_rotas[0])

        return all(len(rota) == rota_length for rota in self.available_rotas)


    def get_rota_length(self) -> int:
        return len(self.available_rotas[0])



    def get_variable_cardinalities(self, ignore_skills=True, ignore_times=True):
        return [len(self.available_rotas), self.get_rota_length()]


class WorkerVariables:
    which_rota: Optional[int]
    starting_day: Optional[int]

    def __init__(self, which_rota: Optional[int], starting_day: Optional[int]):
        self.which_rota = which_rota
        self.starting_day = starting_day



    def effective_rota(self, worker: Worker):
        return worker.available_rotas[self.which_rota].get_rotated_by(starting_day=self.starting_day)




