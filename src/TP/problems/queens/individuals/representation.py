from typing import List, Optional, Tuple

from pydantic import field_validator

from TP.core.individuals.representation import Individual
from TP.problems.queens.fitness import QueensBoardFitness


class QueensIndividual(Individual):
    # chrm: List[int]
    fitness_calculator: QueensBoardFitness
    _fitness: Optional[float] = None
    _unfit_positions: Optional[List[Tuple[int, int]]] = None

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
                'Queens positions are incompatible with board size.'
            )
        return v

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
        for pos, val in enumerate(self.chrm[:-1]):
            for comp_pos in range(pos + 1, len(self.chrm)):
                delta_pos = comp_pos - pos
                comp_val = self.chrm[comp_pos]
                delta_val = abs(val - comp_val)
                if delta_pos == delta_val:
                    incompatible_pair = (pos, comp_pos)
                    incompatible_pos.append(incompatible_pair)
        return incompatible_pos

    def decode(self):
        return self.chrm

    @property
    def fitness(self):
        if self._fitness is None:
            self._fitness = self.fitness_calculator.calc_fitness(self.chrm)
        return self._fitness

    @property
    def unfit_pos(self):
        if self._unfit_positions is None:
            self._unfit_positions = (
                self.fitness_calculator.check_unfit_positions(self.chrm)
            )
        return self._unfit_positions

    def invalidate_fitness(self):
        self._fitness = None

    def invalidate_unfit_pos(self):
        self._unfit_positions = None
