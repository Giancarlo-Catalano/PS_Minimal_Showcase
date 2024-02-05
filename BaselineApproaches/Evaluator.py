from typing import Callable, TypeAlias, Any

from custom_types import Fitness

Individual: TypeAlias = Any
Population: TypeAlias = list[Individual]
Fitness: TypeAlias = float
EvaluatedIndividual: TypeAlias = (Individual, Fitness)
EvaluatedPopulation = list[EvaluatedIndividual]


class Evaluator:
    fitness_function: Callable
    used_evaluations: int

    def __init__(self, fitness_function: Callable):
        self.fitness_function = fitness_function
        self.used_evaluations = 0

    def evaluate(self, individual: Individual) -> Fitness:
        return self.fitness_function(individual)

    def evaluate_population(self, population: Population) -> EvaluatedPopulation:
        self.used_evaluations += len(population)
        return [(individual, self.fitness_function(individual) for individual in population)]


