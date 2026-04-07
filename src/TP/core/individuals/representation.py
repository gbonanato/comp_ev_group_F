from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from TP.core.fitness import FitnessCalculator


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Individual(ABC):
    chrm: List
    fitness_calculator: FitnessCalculator
    _fitness: Optional[float] = None

    @property
    @abstractmethod
    def fitness(self):
        pass

    @abstractmethod
    def decode(self):
        pass


# # REALOCATE
#     fitness_calculator: FitnessCalculator
#     mutation_operator: MutOperator
#     p_m: float = Field(default=0.1, ge=0.0, le=1.0)
