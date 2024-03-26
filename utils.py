import itertools
import random
from typing import Any, Iterable, Callable

import numpy as np
from pandas import DataFrame
import plotly.express as px

import time
from contextlib import ContextDecorator


def first(pair: (Any, Any)) -> Any:
    return pair[0]


def second(pair: (Any, Any)) -> Any:
    return pair[1]


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


def remap_each_column_in_zero_one(input_matrix: np.ndarray) -> np.ndarray:
    result_matrix = np.zeros_like(input_matrix)
    _, columns = input_matrix.shape
    for column in range(columns):
        result_matrix[:, column] = remap_array_in_zero_one(input_matrix[:, column])
    return result_matrix


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def make_interactive_3d_plot(points, labels: list[str]):
    df = DataFrame(data=points, columns=labels)
    fig = px.scatter_3d(df, x=labels[0], y=labels[1], z=labels[2])
    fig.show()


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



def harmonic_mean(values: Iterable[float]) -> float:
    if len(values) == 0:
        raise Exception("Trying to get the harmonic mean of no values!")

    sum_of_inverses = sum(value**(-1) for value in values)
    return (sum_of_inverses/len(sum_of_inverses))**(-1)


def sample_from_geometric_distribution(chance_of_success: float) -> int:
    counter = 0
    while random.random() < chance_of_success:
        counter +=1
    return counter


def get_descriptive_stats(data: np.ndarray) -> (float, float, float, float, float):
    return np.min(data), np.median(data), np.max(data), np.average(data), np.std(data)



def print_entry_and_exit(func):
    def wrapper(*args, **kwargs):
        print(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Terminating {func.__name__}")
        return result
    return wrapper





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


""" Timing example
    with execution_time() as time:
        data = function()
    
    time.execution_time
    print(time)
    print(data)

"""


def repeat(n: int, action: Callable):
    for _ in range(n):
        action()
