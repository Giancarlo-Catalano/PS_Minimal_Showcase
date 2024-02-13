import random
from collections import Counter

import utils
from JMetal.PSProblem import AtomicityEvaluator
from PRef import PRef
from PS import PS
from PSMetric.MeanFitness import MeanFitness
from PSMetric.Simplicity import Simplicity


class SelfAssembly:
    pRef: PRef
    atomicity_evaluator: AtomicityEvaluator
    mean_fitness: MeanFitness


    ps_counter: Counter

    def __init__(self, pRef: PRef):
        self.pRef = pRef
        self.atomicity_evaluator = AtomicityEvaluator(pRef)
        self.simplicity = Simplicity()
        self.mean_fitness = MeanFitness()

        self.ps_counter = Counter()

    def tournament_select_alternative(self, vars_and_scores: list[(PS, float)]) -> PS:
        tournament_size = 3
        tournament = random.choices(vars_and_scores, k=tournament_size)
        return max(tournament, key=utils.second)[0]

    def get_atomicity(self, ps: PS) -> float:
        return self.atomicity_evaluator.evaluate_single(ps)

    def get_mean_fitness(self, ps: PS) -> float:
        return self.mean_fitness.get_single_score(ps, self.pRef)

    @property
    def search_space(self):
        return self.pRef.search_space

    def improvement_step(self, ps: PS) -> PS:
        chance_of_simplification = ps.fixed_count() / len(ps)

        alternatives = None
        if random.random() < chance_of_simplification:
            alternatives = [(simpl, self.get_atomicity(simpl))
                            for simpl in ps.simplifications()]
        else:
            alternatives = [(spec, self.get_mean_fitness(spec))
                            for spec in ps.specialisations(self.search_space)]

        return self.tournament_select_alternative(alternatives)

    def improve_continously(self, ps: PS) -> PS:
        iterations = self.search_space.hot_encoded_length*2
        current = PS(ps.values)
        for iteration in range(iterations):
            current = self.improvement_step(current)
            self.ps_counter.update([current])
            #print(f"\t{current}")

        return current

    def select_random_fixed_start(self) -> PS:
        tournament_size = 30
        tournament_indexes = random.choices(list(enumerate(self.pRef.fitness_array)), k=tournament_size)

        tournament = [(self.pRef.full_solutions[index], fitness) for index, fitness in tournament_indexes]
        winner_full_solution = max(tournament, key=utils.second)[0]
        return PS.from_FS(winner_full_solution)

    def unfixed_start(self) -> PS:
        return PS.empty(self.search_space)
