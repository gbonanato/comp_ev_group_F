from typing import List, Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from TP.core.individuals.encoding import Encoder
from TP.core.multiobjective.fitness import MOFitnessCalculator, ProblemMetric


@dataclass(eq=False, config=ConfigDict(arbitrary_types_allowed=True))
class MOIndividual:
    chrm: List
    encoder: Encoder
    fitness_calculator: MOFitnessCalculator
    _fitness: Optional[ProblemMetric] = None

    def __post_init__(self):
        self.encoder.validate(self.chrm)

    @property
    def fitness(self) -> ProblemMetric:
        if self._fitness is None:
            self._fitness = self.fitness_calculator.calc_fitness(self.chrm)
        return self._fitness

    def decode(self):
        return self.encoder.decode(self.chrm)

    # def __hash__(self):
    #     return hash(tuple(self.chrm))

    # def __eq__(self, other):
    #     if not isinstance(other, MOIndividual):
    #         return False
    #     return self.chrm == other.chrm
