import math
from typing import TypeAlias

import numpy as np

from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.RotaPattern import RotaPattern, get_range_scores
from BenchmarkProblems.BT.Worker import Worker, Skill
from Core.FullSolution import FullSolution

ExtendedPattern: TypeAlias = np.ndarray


def rota_to_extended_pattern(rota: RotaPattern, calendar_length: int) -> ExtendedPattern:
    pattern = np.array([day.working for day in rota.days], dtype=int)
    if len(pattern) >= calendar_length:
        return pattern[:calendar_length]

    return np.tile(pattern, math.ceil(calendar_length / len(pattern)))[:calendar_length]


def get_rotated_by_starting_week(full_pattern: ExtendedPattern, starting_week: int) -> ExtendedPattern:
    return np.roll(full_pattern, -starting_week)


def convert_worker_to_just_options(worker: Worker, calendar_length: int) -> np.ndarray:
    return np.array([rota_to_extended_pattern(rota, calendar_length)
                     for rota in worker.available_rotas])



FullPatternOptions: TypeAlias = np.ndarray
DayRange: TypeAlias = float
WeekRanges: TypeAlias = np.ndarray

class EfficientBTProblem(BTProblem):
    extended_patterns: list[FullPatternOptions]
    workers_by_skills: dict  # Skill -> set[worker index]

    def __init__(self,
                 workers: list[Worker],
                 calendar_length: int):
        super().__init__(workers, calendar_length)
        self.extended_patterns = [convert_worker_to_just_options(worker, calendar_length)
                                  for worker in workers]
        self.workers_by_skills = {skill: {index for index, worker in enumerate(self.workers)
                                          if skill in worker.available_skills}
                                  for skill in self.all_skills}

    def get_ranges_for_weekdays_for_skill(self, chosen_patterns: list[ExtendedPattern], skill: Skill) -> WeekRanges:
        indexes = self.workers_by_skills[skill]
        summed_patterns: ExtendedPattern = sum(chosen_patterns[index] for index in indexes)  # not np.sum because it doesn't support generators
        summed_patterns = summed_patterns.reshape((-1, 7))
        return get_range_scores(summed_patterns)

    def aggregate_range_scores(self, range_scores: WeekRanges) -> float:
        return float(np.sum(day_range * weight for day_range, weight in zip(range_scores, self.weights)))


    def get_chosen_patterns_from_fs(self, fs: FullSolution) -> list[ExtendedPattern]:
        return [options[which] for options, which in zip(self.extended_patterns, fs.values)]


    def fitness_function(self, fs: FullSolution) -> float:
        chosen_patterns = self.get_chosen_patterns_from_fs(fs)
        total_score = np.sum(self.aggregate_range_scores(self.get_ranges_for_weekdays_for_skill(chosen_patterns, skill))
                             for skill in self.all_skills)
        return -total_score

