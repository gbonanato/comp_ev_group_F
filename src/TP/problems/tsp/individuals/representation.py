from typing import List, Optional, Tuple

from pydantic import field_validator

from TP.core.individuals.representation import Individual
from TP.problems.tsp.fitness import TSPFitness


class TSPIndividual(Individual):
    # chrm: List[int]
    fitness_calculator: TSPFitness
    _fitness: Optional[float] = None

    @field_validator('chrm')
    @classmethod
    def check_uniqueness(cls, v):
        if len(v) != len(set(v)):
            raise ValueError('Chromossome does not contains unique values.')
        return v

    @field_validator('chrm')
    @classmethod
    def check_n_value(cls, v):
        if max(v) != len(set(v)):
            raise ValueError(
                'Path lenght incompatible with nodes number.'
            )
        return v
