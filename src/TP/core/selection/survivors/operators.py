import heapq
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import Field
from pydantic.dataclasses import dataclass

from TP.core.individuals.population import Population
from TP.core.individuals.representation import Individual


@dataclass
class SurvivorSelector(ABC):
    @abstractmethod
    def select_survivors(
        parents: List[Individual],
        offsprings: List[Individual],
        n_survivors: Optional[int],
    ) -> Population:
        pass


@dataclass
class Generational(SurvivorSelector):
    def select_survivors(parents, offsprings) -> Population:
        return Population(offsprings)


@dataclass
class ElitismGenerational(SurvivorSelector):
    elite_pop_pct: float = Field(default=0.1, ge=0, le=1)
    distinct: bool = True

    def select_survivors(
        self,
        parents: List[Individual],
        offsprings: List[Individual],
        n_survivors: int,
    ) -> Population:

        elite_pop_size = int(self.elite_pop_pct * n_survivors)

        # Combine parents and offsprings for global elitism
        candidates = parents + offsprings
        sorted_candidates = heapq.nlargest(
            len(candidates), candidates, key=lambda ind: ind.fitness
        )

        elites = []
        seen = set()

        for ind in sorted_candidates:
            if len(elites) >= elite_pop_size:
                break

            if self.distinct:
                chrm_key = tuple(ind.chrm)
                if chrm_key in seen:
                    continue
                seen.add(chrm_key)

            elites.append(ind)

        # Fill remaining slots (offsprings prioritized, fitness-based)
        remaining = n_survivors - len(elites)

        rest_population = heapq.nlargest(
            remaining,
            offsprings,
            key=lambda ind: ind.fitness,
        )

        next_pop = elites + rest_population
        return Population(next_pop)
