from dataclasses import dataclass
from typing import List

import numpy as np

from TP.core.fitness import FitnessCalculator

epsilon = 1e-6


@dataclass
class TSPFitness(FitnessCalculator):
    problem_instance: np.ndarray

    def calc_fitness(
        self,
        chrm: List[int],
    ):

        total_distance = 0
        total_distance += self.problem_instance[
            0,  # Tour start
            chrm[0],
        ]
        for i in range(len(chrm) - 1):
            total_distance += self.problem_instance[
                chrm[i],
                chrm[i + 1],
            ]

        total_distance += self.problem_instance[
            chrm[-1],
            0,  # Tour end
        ]

        fitness = 1 / (total_distance + epsilon)

        return fitness
