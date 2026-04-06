import heapq
from abc import ABC, abstractmethod
from typing import List

from pydantic import Field

from TP.core.individuals.representation import Individual


class SurvivorSelector(ABC):
    @abstractmethod
    def select_survivors(
        parents: List[Individual],
        offsprings: List[Individual],
        n_survivors: int,
    ):
        pass


class Generational(SurvivorSelector):
    def select_survivors(parents, offsprings):
        return offsprings


class ElitismGenerational(SurvivorSelector):
    elite_pop_pct: float = Field(le=0, ge=1, default=0.1)

    def select_survivors(
        self,
        parents: List[Individual],
        offsprings: List[Individual],
        n_survivors: int,
    ):
        elite_pop_size = int(self.elite_pop_pct * n_survivors)
        elite_parents = heapq.nlargest(
            elite_pop_size, parents, key=lambda ind: ind.fitness
        )
        remaining = n_survivors - elite_pop_size

        sorted_offsprings = heapq.nlargest(
            remaining, offsprings, key=lambda ind: ind.fitness
        )

        return elite_parents + sorted_offsprings
