from typing import List

import numpy as np

from TP.core.fitness import FitnessCalculator


class TSPFitness(FitnessCalculator):
    def calc_fitness(self, chrm: List[int], dist_matrix: np.ndarray):
        cost = 0
        for pos in range(2, len(chrm) + 1):
            i, j = chrm[pos - 2 : pos]
            cost += dist_matrix[i][j]
        return cost
