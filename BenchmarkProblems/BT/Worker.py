from BenchmarkProblems.BT.RotaPattern import RotaPattern


class Skill:
    skill_index: int


    def __init__(self, skill_index: int)
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


    def __repr__(self):
        return self.name



