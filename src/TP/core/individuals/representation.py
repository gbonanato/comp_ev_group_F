from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field

from TP.core.fitness import FitnessCalculator
from TP.core.variation.mutation import MutOperator


class Individual(BaseModel, ABC):
    chrm: List
    fitness_calculator: FitnessCalculator
    mutation_operator: MutOperator
    p_m: float = Field(default=0.1, ge=0.0, le=1.0)
    _fitness: Optional[float] = None

    @abstractmethod
    def mutate(self):
        pass

    @property
    def fitness(self):
        pass

    @abstractmethod
    def decode(self):
        pass
