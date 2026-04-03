from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field, field_validator

from TP.queen_problem.utils.variation.mutation import MutOperator


class Individual(BaseModel, ABC):
    chrm: List
    mutation_operator: MutOperator
    p_m: float = Field(ge=0.0, le=1.0)

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def calc_fitness(self):
        pass


class PermIndividual(Individual):
    chrm: List[int]

    @field_validator('chrm')
    @classmethod
    def check_uniqueness(cls, v):
        if len(v) != len(set(v)):
            raise ValueError('Chromossome does not contains unique values.')

    def mutate():
        pass

    def check_unfit_positions(self) -> List[tuple[int, int]]:
        """
        Check the incompatible queen positions on the soution
        candidate. If the column distance is equal to the rows distance
        between the queens, then the positions are considered incompatible.

        Returns
        -------
        List[tuple[int, int]]
            list of tuple of pairs of incompatible positions.
        """
        incompatible_pos = []
        for pos, val in enumerate(self.chrm):
            for comp_pos, comp_val in range(pos, len(self.chrm) + 1):
                delta_pos = comp_pos - pos
                delta_val = abs(val - comp_val)
                if delta_pos == delta_val:
                    incompatible_pair = (pos, comp_pos)
                    incompatible_pos.append(incompatible_pair)
        return incompatible_pos

    def calc_fitness(self, penalty=-10):
        incompatible_pos = self.check_unfit_positions()
        fitness = len(incompatible_pos) * penalty
        return fitness
