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

    def select_survivors(
        self,
        parents: List[Individual],
        offsprings: List[Individual],
        n_survivors: int,
    ) -> Population:
        elite_pop_size = int(self.elite_pop_pct * n_survivors)
        elite_parents = heapq.nlargest(
            elite_pop_size, parents, key=lambda ind: ind.fitness
        )
        remaining = n_survivors - elite_pop_size

        sorted_offsprings = heapq.nlargest(
            remaining, offsprings, key=lambda ind: ind.fitness
        )

        next_pop = elite_parents + sorted_offsprings

        return Population(next_pop)
