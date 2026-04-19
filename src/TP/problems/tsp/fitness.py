from dataclasses import dataclass
from typing import List

from tsplib95.models import StandardProblem

from TP.core.fitness import FitnessCalculator

epsilon = 1e-6


@dataclass
class TSPFitness(FitnessCalculator):
    problem_instance: StandardProblem

    def calc_fitness(
        self,
        chrm: List[int],
    ):

        total_distance = 0
        total_distance += self.problem_instance.get_weight(  # Tour start
            1, chrm[0]
        )
        for i in range(len(chrm) - 2):
            total_distance += self.problem_instance.get_weight(
                chrm[i], chrm[i + 1]
            )

        total_distance += self.problem_instance.get_weight(  # Tour end
            chrm[-1], 1
        )

        fitness = 1 / (total_distance + epsilon)

        return fitness
