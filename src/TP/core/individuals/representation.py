from typing import List, Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from TP.core.fitness import FitnessCalculator
from TP.core.individuals.encoding import Encoder


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class Individual:
    chrm: List
    encoder: Encoder
    fitness_calculator: FitnessCalculator
    _fitness: Optional[float] = None

    def __post_init__(self):
        self.encoder.validate(self.chrm)

    @property
    def fitness(self):
        if self._fitness is None:
            self._fitness = self.fitness_calculator.calc_fitness(self.chrm)
        return self._fitness

    def decode(self):
        return self.encoder.decode(self.chrm)
