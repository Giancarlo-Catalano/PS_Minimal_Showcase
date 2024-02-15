import random
from collections import Counter

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from PRef import PRef
from PS import PS
from PSMetric.Atomicity import Atomicity
from PSMetric.MeanFitness import MeanFitness


class SelfAssembly:
    pRef: PRef

    mean_fitness: MeanFitness
    atomicity: Atomicity

    ps_counter: Counter

    def __init__(self, pRef: PRef):
        self.mean_fitness = MeanFitness()
        self.atomicity = Atomicity()
        self.mean_fitness.set_pRef(pRef)
        self.atomicity.set_pRef(pRef)

        self.ps_counter = Counter()

    def tournament_select_alternative(self, vars_and_scores: list[(PS, float)]) -> PS:
        tournament_size = 3
        tournament = random.choices(vars_and_scores, k=tournament_size)
        return max(tournament, key=utils.second)[0]

    @property
    def search_space(self):
        return self.pRef.search_space

    def improvement_step(self, ps: PS) -> PS:
        chance_of_simplification = ps.fixed_count() / len(ps)

        alternatives = None
        if random.random() < chance_of_simplification:
            alternatives = [(simpl, self.atomicity.get_single_score(simpl))
                            for simpl in ps.simplifications()]
        else:
            alternatives = [(spec, self.mean_fitness.get_single_score(spec))
                            for spec in ps.specialisations(self.search_space)]

        return self.tournament_select_alternative(alternatives)

    def improve_continously(self, ps: PS) -> PS:
        iterations = self.search_space.hot_encoded_length * 2
        current = PS(ps.values)
        for iteration in range(iterations):
            current = self.improvement_step(current)
            self.ps_counter.update([current])
            # print(f"\t{current}")

        return current

    def select_random_fixed_start(self) -> PS:
        tournament_size = 30
        tournament_indexes = random.choices(list(enumerate(self.pRef.fitness_array)), k=tournament_size)

        tournament = [(self.pRef.full_solutions[index], fitness) for index, fitness in tournament_indexes]
        winner_full_solution = max(tournament, key=utils.second)[0]
        return PS.from_FS(winner_full_solution)

    def unfixed_start(self) -> PS:
        return PS.empty(self.search_space)


def test_simple_hill_climber(problem: BenchmarkProblem):
    sa = SelfAssembly(problem.get_pRef(10000))

    for _ in range(60):
        if random.random() < 0.5:
            starting_point = sa.select_random_fixed_start()
        else:
            starting_point = sa.unfixed_start()

        final = sa.improve_continously(starting_point)

        print(f"Starting from {starting_point}, we got to {final}")

    print("At the end, the top 12 best partial solutions are")
    for ps, count in sa.ps_counter.most_common(12):
        print(f"{ps}, appearing {count} times")
