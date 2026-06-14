from abc import ABC
from typing import List, Optional

from pydantic.dataclasses import dataclass

from TP.core.individuals.representation import Individual


@dataclass
class Population(ABC):
    ind_list: List[Individual]
    _select_prob_cache: Optional[List[float]] = None

    def __iter__(self):
        return iter(self.ind_list)

    @property
    def size(self):
        return len(self.ind_list)

    def get_individuals_fitness(self):
        return [i.fitness for i in self.ind_list]
