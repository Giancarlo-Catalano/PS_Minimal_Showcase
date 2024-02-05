import unittest

from FullSolution import FullSolution
from PSMetric.Atomicity import Atomicity
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity
from PRef import PRef
from PS import PS
from SearchSpace import SearchSpace
from custom_types import Fitness

if __name__ == '__main__':
    ps: PS = PS([1, -1, 0, -1])

    print(f"The PS is {ps}")

    search_space = SearchSpace([2, 2, 2, 2])
    def fitness_function(fs: FullSolution) -> Fitness:
        return fs.values.sum(dtype=float)


    pRef: PRef = PRef.sample_from_search_space(search_space, fitness_function, 400)

    # print(f"The pRef is {pRef.long_repr()}")

    simplicity = Simplicity()
    mean_fitness = MeanFitness()
    atomicity = Atomicity()


    simplicity_score = simplicity.get_single_unnormalised_score(ps, pRef)
    mean_fitness_score = mean_fitness.get_single_unnormalised_score(ps, pRef)
    atomicity_score = atomicity.get_unnormalised_scores([ps], pRef)

    print(f"The obtained scores for the PS are simplicity = {simplicity_score},"
          f"\nand mean_fitness = {mean_fitness_score},"
          f"\nand atomicity = {atomicity_score}")
