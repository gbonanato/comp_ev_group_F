from typing import List

from TP.core.fitness import FitnessCalculator


class QueensBoardFitness(FitnessCalculator):
    @staticmethod
    def check_unfit_positions(
        chrm: List[int],
    ) -> List[tuple[int, int]]:
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
        for pos, val in enumerate(chrm[:-1]):
            for comp_pos in range(pos + 1, len(chrm)):
                delta_pos = comp_pos - pos
                comp_val = chrm[comp_pos]
                delta_val = abs(val - comp_val)
                if delta_pos == delta_val:
                    incompatible_pair = (pos, comp_pos)
                    incompatible_pos.append(incompatible_pair)
        return incompatible_pos

    def calc_fitness(self, chrm: List[int], weight: float = -1):
        unfit_positions = self.check_unfit_positions(chrm)
        return len(unfit_positions) * weight
