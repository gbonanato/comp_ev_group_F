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
        # if chrm[0] != 1 and chrm[-1] != 1: Se isso vai ser add em todos...
        # não faz diferença para achar o melhor!!
        #     chrm.insert(0, 1)  # Ensures cicle starts
        #     chrm.append(1)  # and ends on 1
        total_distance = sum(
            self.problem_instance.get_weight(chrm[i], chrm[i + 1])
            for i in range(len(chrm) - 1)
        )
        fitness = 1 / (total_distance + epsilon)

        return fitness
