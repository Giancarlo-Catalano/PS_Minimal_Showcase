from typing import Optional


class WorkDay:
    working: bool
    start_time: Optional[int]
    end_time: Optional[int]


    def __init__(self, working: bool, start_time: Optional[int], end_time: Optional[int]):
        self.working = working
        self.start_time = start_time
        self.end_time = end_time


    @classmethod
    def working_day(cls, start_time, end_time):
        return cls(True, start_time, end_time)

    @classmethod
    def not_working(cls):
        return cls(False, None, None)

    def __repr__(self):
        if self.working:
            return f"{self.start_time}~{self.end_time}"
        else:
            return "----"

class RotaPattern:
    workweek_length: int
    days: list[WorkDay]

    def __init__(self, workweek_length: int, days: list[WorkDay]):
        assert(len(days) % workweek_length == 0)
        self.workweek_length = workweek_length
        self.days = days


    def __repr__(self):
        split_by_week = [self.days[(which*self.workweek_length):(which+1*self.workweek_length)]
                          for which in range(len(self.days) // self.workweek_length)]

        def repr_week(week: list[WorkDay]) -> str:
            return "<" + ", ".join(f"{day}" for day in week)+">"

        return ", ".join(map(repr_week, split_by_week))

    def get_rotated_by(self, starting_day: int):
        assert(starting_day < len(self.days))
        return self.days[starting_day:]+self.days[:starting_day]




