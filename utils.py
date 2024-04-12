import time
from contextlib import ContextDecorator
from typing import Iterable, Any

import numpy as np


def unzip(zipped):
    if len(zipped) == 0:
        return []

    group_amount = len(zipped[0])

    def get_nth_group(n):
        return [elem[n] for elem in zipped]

    return tuple(get_nth_group(n) for n in range(group_amount))


def remap_array_in_zero_one(input_array: np.ndarray):
    """remaps the values in the given array to be between 0 and 1"""
    min_value = np.min(input_array)
    max_value = np.max(input_array)

    if min_value == max_value:
        return np.full_like(input_array, 0.5, dtype=float)  # all 0.5!

    return (input_array - min_value) / (max_value - min_value)


def harmonic_mean(values: Iterable[float]) -> float:
    if len(values) == 0:
        raise Exception("Trying to get the harmonic mean of no values!")

    sum_of_inverses = sum(value ** (-1) for value in values)
    return (sum_of_inverses / len(sum_of_inverses)) ** (-1)


def get_descriptive_stats(data: np.ndarray) -> (float, float, float, float, float):
    return np.min(data), np.median(data), np.max(data), np.average(data), np.std(data)


class ExecutionTime(ContextDecorator):
    start_time: float
    end_time: float
    execution_time: float

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time

    def __str__(self):
        return f"{self.execution_time:.6f}"


def execution_time():
    return ExecutionTime()


class Announce(ContextDecorator):
    action_str: str
    timer: ExecutionTime

    def __init__(self, action_str: str):
        self.action_str = action_str
        self.timer = ExecutionTime()

    def __enter__(self):
        print(self.action_str, end="...")
        self.timer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.__exit__(exc_type, exc_val, exc_tb)
        runtime = self.timer.execution_time
        print(f"...Finished (took {runtime:2f} seconds)")


def announce(action: str):
    return Announce(action)


""" Timing example
    with execution_time() as time:
        data = function()
    
    time.execution_time
    print(time)
    print(data)

"""


def indent(input: str) -> str:
    lines = input.split("\n")
    lines = ["\t" + line for line in lines]
    return "\n".join(lines)


def break_list(input_list: list[Any], group_size: int) -> list[list[Any]]:
    def start(which):
        return group_size * which

    def end(which):
        return group_size * (which + 1)

    return [input_list[start(i):end(i)] for i in range(len(input_list) // group_size)]


def join_lists(many_lists: Iterable[list]) -> list:
    result = []
    for sub_list in many_lists:
        result.extend(sub_list)

    return result