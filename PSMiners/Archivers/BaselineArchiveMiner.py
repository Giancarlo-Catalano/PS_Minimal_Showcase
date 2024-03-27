import random

from PRef import PRef
from PS import PS
from PSMetric.Metric import Metric
from PSMiners.Individual import Individual
from PSMiners.PSMiner import PSMiner, Population


# performance issues:

class BaselineArchiveMiner(PSMiner):
    """use this for testing"""
    population_size: int
    offspring_population_size: int

    archive: set[Individual]

    def __init__(self,
                 population_size: int,
                 offspring_population_size: int,
                 metric: Metric,
                 pRef: PRef):
        self.population_size = population_size
        self.offspring_population_size = offspring_population_size

        super().__init__(metric=metric,
                         pRef=pRef)
        self.archive = set()


    def __repr__(self):
        return f"ArchiveMiner(population_size = {self.population_size}, offspring_size = {self.offspring_population_size})"

    def get_initial_population(self) -> Population:
        """ basically takes the elite of the PRef, and converts them into PSs """
        """this is called get_init in the paper"""
        return [Individual(PS.empty(self.search_space))]

    def get_localities(self, individual: Individual) -> list[Individual]:
        return [Individual(ps) for ps in individual.ps.specialisations(self.search_space)]

    def select_one(self) -> Individual:
        tournament_size = 3
        tournament_pool = random.choices(self.current_population, k=tournament_size)
        return max(tournament_pool, key=lambda x: x.aggregated_score)

    def step(self):
        """Note how this relies on the current population being evaluated,
        and the new population will also be evaluated"""
        # select and add to archive, so that they won't appear in the population again
        offspring = set()

        remaining_population = set(self.current_population)
        while len(offspring) < self.offspring_population_size and len(remaining_population) > 0:
            selected = self.select_one()
            remaining_population.discard(selected)
            if selected not in self.archive:
                self.archive.add(selected)
                offspring.update(self.get_localities(selected))

        self.current_population = list(remaining_population)
        self.current_population.extend(self.evaluate_individuals(list(offspring)))
        self.current_population = self.get_best_n(self.population_size, self.current_population)

    def get_results(self, quantity_returned: int) -> Population:
        return self.get_best_n(n=quantity_returned, population=list(self.archive))

    def get_parameters_as_dict(self) -> dict:
        return {"kind": "Archive",
                "population_size": self.population_size,
                "offspring_size": self.offspring_population_size,
                "metric": repr(self.metric)}
