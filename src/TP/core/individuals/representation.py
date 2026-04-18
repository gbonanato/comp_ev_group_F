from abc import ABC
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
    def fitness(self):
        if self._fitness is None:
            self._fitness = self.fitness_calculator.calc_fitness(self.chrm)
        return self._fitness

    def decode(self):
        return self.chrm


# # REALOCATE
#     fitness_calculator: FitnessCalculator
#     mutation_operator: MutOperator
#     p_m: float = Field(default=0.1, ge=0.0, le=1.0)
