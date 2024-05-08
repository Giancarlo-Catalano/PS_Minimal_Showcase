import itertools
import math
from typing import TypeAlias

import numpy as np

from BenchmarkProblems.BT.BTProblem import BTProblem
from BenchmarkProblems.BT.RotaPattern import RotaPattern, get_range_scores
from BenchmarkProblems.BT.Worker import Worker, Skill
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from Core.custom_types import JSON

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

class CohortMember:
    worker: Worker
    chosen_rota_index: int
    chosen_rota_entended: ExtendedPattern


    def __init__(self,
                 worker: Worker,
                 rota_index: int,
                 calendar_length: int):
        self.worker = worker
        self.chosen_rota_index = rota_index
        self.chosen_rota_extended = rota_to_extended_pattern(rota=worker.available_rotas[rota_index],
                                                             calendar_length=calendar_length)

    def to_json(self) -> JSON:
        return {"worker": self.worker.to_json(),
                "chosen_rota": int(self.chosen_rota_index)}

    @classmethod
    def from_json(cls, element: JSON):
        chosen_rota = int(element["chosen_rota"])
        calendar_length = 91  # TODO fetch this from somewhere else
        worker = Worker.from_json(element["worker"])
        return cls(worker=worker, rota_index=chosen_rota, calendar_length=calendar_length)


    def get_amount_of_skills(self) -> int:
        return len(self.worker.available_skills)

    def get_amount_of_working_hours(self) -> int:
        return int(np.sum(self.chosen_rota_extended))


    def get_amount_of_choices(self) -> int:
        return len(self.worker.available_rotas)


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


def get_amount_of_shared_skills(cohort: Cohort) -> int:
    if len(cohort) == 0:
        return 0

    skillsets = [component.worker.available_skills for component in cohort]
    common_to_all = set.intersection(*skillsets)
    return len(common_to_all)


def get_skill_variation(cohort: Cohort) -> float:
    all_skills = set(skill for component in cohort
                          for skill in component.worker.available_skills)
    sum_of_available_skills = sum(len(component.worker.available_skills) for component in cohort)
    return len(all_skills) / sum_of_available_skills


def get_hamming_distances(cohort: Cohort) -> list[int]:

    def hamming_distance(component_a: CohortMember, component_b: CohortMember) -> int:
        rota_a = component_a.chosen_rota_extended
        rota_b = component_b.chosen_rota_extended

        return int(np.sum(rota_a != rota_b))

    if len(cohort) == 2:  # makes my life a lot easier for data analysis
        distance = hamming_distance(cohort[0], cohort[1])
        return [distance, distance]

    return [hamming_distance(a, b)
               for a, b in itertools.combinations(cohort, 2)]


def get_ranges_in_weekdays(cohort: Cohort) -> np.ndarray:
    total_pattern: np.ndarray = np.array(sum(member.chosen_rota_extended for member in cohort))
    total_pattern = total_pattern.reshape((-1, 7))

    def range_score(min_amount, max_amount):
        if max_amount == 0:
            return 0
        return (max_amount - min_amount) / max_amount
    def range_for_column(column_index: int) -> float:
        values = total_pattern[:, column_index]
        min_value = min(values)
        max_values = max(values)
        return range_score(min_value, max_values)

    return np.array([range_for_column(i) for i in range(7)])

class EfficientBTProblem(BTProblem):
    extended_patterns: list[FullPatternOptions]
    workers_by_skills: dict  # Skill -> set[worker index]
    use_faulty_fitness_function: bool

    def __init__(self,
                 workers: list[Worker],
                 calendar_length: int,
                 use_faulty_fitness_function: bool = False):
        super().__init__(workers, calendar_length)
        self.extended_patterns = [convert_worker_to_just_options(worker, calendar_length)
                                  for worker in workers]
        self.workers_by_skills = {skill: {index for index, worker in enumerate(self.workers)
                                          if skill in worker.available_skills}
                                  for skill in self.all_skills}
        self.use_faulty_fitness_function = use_faulty_fitness_function

    def get_ranges_for_weekdays_for_skill(self, chosen_patterns: list[ExtendedPattern], skill: Skill) -> WeekRanges:
        indexes = self.workers_by_skills[skill]
        summed_patterns: ExtendedPattern = sum(chosen_patterns[index] for index in indexes)  # not np.sum because it doesn't support generators
        summed_patterns = summed_patterns.reshape((-1, 7))
        return get_range_scores(summed_patterns)

    def aggregate_range_scores(self, range_scores: WeekRanges) -> float:
        return float(np.sum(day_range * weight for day_range, weight in zip(range_scores, self.weights)))

    def faulty_aggregate_range_scores(self, range_scores: WeekRanges) -> float:
        # this is the fault
        for index, day_range in enumerate(range_scores):
            if day_range == 1:
                range_scores[index]=0

        return float(np.sum(day_range * weight for day_range, weight in zip(range_scores, self.weights)))


    def get_chosen_patterns_from_fs(self, fs: FullSolution) -> list[ExtendedPattern]:
        return [options[which] for options, which in zip(self.extended_patterns, fs.values)]


    def fitness_function(self, fs: FullSolution) -> float:
        if self.use_faulty_fitness_function:
            return self.faulty_fitness_function(fs)
        else:
            return self.correct_fitness_function(fs)


    def correct_fitness_function(self, fs: FullSolution) -> float:
        chosen_patterns = self.get_chosen_patterns_from_fs(fs)
        total_score = np.sum(self.aggregate_range_scores(self.get_ranges_for_weekdays_for_skill(chosen_patterns, skill))
                             for skill in self.all_skills)
        return -total_score


    def faulty_fitness_function(self, fs: FullSolution) -> float:
        chosen_patterns = self.get_chosen_patterns_from_fs(fs)
        total_score = np.sum(self.faulty_aggregate_range_scores(self.get_ranges_for_weekdays_for_skill(chosen_patterns, skill))
                             for skill in self.all_skills)   # NOTE THAT we're using the faulty one
        return -total_score


    def ps_to_properties(self, ps: PS) -> dict:
        cohort = ps_to_cohort(self, ps)

        mean_rota_choice_amount = np.average([member.get_amount_of_choices() for member in cohort])
        mean_amount_of_hours = np.average([member.get_amount_of_working_hours() for member in cohort])
        mean_hamming_distance = np.average(get_hamming_distances(cohort))
        local_fitness = np.average(get_ranges_in_weekdays(cohort))

        return {"mean_rota_choice_quantity": mean_rota_choice_amount,
                "mean_amount_of_hours": mean_amount_of_hours,
                "mean_difference_in_rotas": mean_hamming_distance,
                "local_fitness": local_fitness}

    def repr_property(self, property_name:str, property_value:str, property_rank_range:str):
        lower_rank, upper_rank = property_rank_range
        is_low = lower_rank < 0.5
        if property_name == "mean_rota_choice_quantity":
            return f"They have relatively {'few' if is_low else 'many'} rota choices (mean = {property_value:.2f})"
        elif property_name == "mean_amount_of_hours":
            return f"The working hour amounts are {'low' if is_low else 'high'} (mean = {property_value:.2f})"
        elif property_name == "mean_difference_in_rotas":
            return f"The rotas are generally {'similar' if is_low else 'different'} (rank = {int(property_rank_range[0]*100)}% ~ {int(property_rank_range[1]*100)}%)"
        elif property_name == "local_fitness":
            return f"The rotas {'' if is_low else 'do not '}complement each other (rank = {int(property_rank_range[0]*100)}% ~ {int(property_rank_range[1]*100)}%)"

