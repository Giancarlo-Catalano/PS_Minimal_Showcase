import itertools
import random
from typing import TypeAlias

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FullSolution import FullSolution
from SearchSpace import SearchSpace

Node: TypeAlias = int
Connection: TypeAlias = (Node, Node)

class GraphColouring(BenchmarkProblem):
    amount_of_colours: int
    amount_of_nodes: int

    connections: list[Connection]


    def __init__(self,
                 amount_of_colours: int,
                 amount_of_nodes: int,
                 connections: list[Connection]):
        self.amount_of_colours = amount_of_colours
        self.amount_of_nodes = amount_of_nodes
        self.connections = connections

        search_space = SearchSpace([amount_of_colours for node in range(self.amount_of_nodes)])
        super().__init__(search_space)


    def __repr__(self):
        return f"GraphColouring(#colours = {self.amount_of_colours}, #nodes = {self.amount_of_nodes})"

    def long_repr(self) -> str:
        return self.__repr__()+"  "+", ".join(f"{connection}" for connection in self.connections)

    @classmethod
    def random(cls, amount_of_nodes: int,
                    amount_of_colours: int,
                    chance_of_connection: float):
        connections = []
        for node_a, node_b in itertools.combinations(range(amount_of_nodes), 2):
            if random.random() < chance_of_connection:
                connections.append((node_a, node_b))

        return cls(amount_of_colours = amount_of_colours,
                   amount_of_nodes = amount_of_nodes,
                   connections = connections)

    def fitness_function(self, fs: FullSolution) -> float:
        return float(sum([1 for (node_a, node_b) in self.connections
                          if fs.values[node_a] != fs.values[node_b]]))