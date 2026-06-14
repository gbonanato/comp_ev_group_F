import math
from dataclasses import dataclass
from typing import List, NamedTuple

from TP.core.multiobjective.fitness import (
    MOFitnessCalculator,
    Objective,
)
from TP.problems.vrp.utils.best_cuts import DPFindCuts
from TP.problems.vrp.utils.problem_data import VRPProblemData

epsilon = 1e-6


class RouteMetrics(NamedTuple):
    total_distance: Objective
    max_imbalance: Objective


@dataclass
class VRPFitness(MOFitnessCalculator):
    problem_instance: VRPProblemData

    def calc_fitness(
        self,
        chrm: List[int],
    ) -> RouteMetrics:

        route_spliter = DPFindCuts()
        vec_dist_dict = route_spliter.split_route(chrm, self.problem_instance)
        total_dist = 0
        min_dist = math.inf
        max_dist = -math.inf

        for _, dist in vec_dist_dict.items():
            total_dist += dist
            min_dist = min(min_dist, dist)
            max_dist = max(max_dist, dist)

        max_imbalance = max_dist - min_dist
        total_dist_obj = Objective(
            name='total_distance',
            value=total_dist,
            higher_is_better=False,
        )
        max_imbalance_obj = Objective(
            name='max_imbalance',
            value=max_imbalance,
            higher_is_better=False,
        )
        fitness = RouteMetrics(
            total_distance=total_dist_obj,
            max_imbalance=max_imbalance_obj,
        )

        return fitness
