import math
from dataclasses import dataclass
from typing import List, NamedTuple

from TP.core.multiobjective.fitness import MOFitnessCalculator
from TP.problems.vrp.utils.best_cuts import DPFindCuts
from TP.problems.vrp.utils.problem_data import CVRPProblemData

epsilon = 1e-6


class RouteMetrics(NamedTuple):
    total_distance: float
    max_imbalance: float


@dataclass
class CVRPFitness(MOFitnessCalculator):
    problem_instance: CVRPProblemData
    route_spliter: DPFindCuts

    def calc_fitness(
        self,
        chrm: List[int],
    ):

        vec_dist_dict = self.route_spliter.split_route(
            chrm, self.problem_instance
        )
        total_dist = 0
        min_dist = math.inf
        max_dist = -math.inf

        for vec, dist in vec_dist_dict.items():
            total_dist += dist
            min_dist = min(min_dist, dist)
            min_dist = max(max_dist, dist)

        max_imbalance = max_dist - min_dist
        fitness = RouteMetrics(
            total_distance=total_dist,
            max_imbalance=max_imbalance,
        )

        return fitness
