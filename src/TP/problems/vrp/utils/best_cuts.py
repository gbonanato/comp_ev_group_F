import heapq
import math
from typing import Dict, List, Tuple

import numpy as np
from pydantinc.dataclasses import dataclass

from TP.problems.vrp.utils.problem_data import CVRPProblemData


@dataclass
class DPFindCuts:
    """
    This code uses dynamic programming to find the shortest path in a DAG.
    Since the tour is provided as a giant tour, this code finds the optimal
    cuts to minimize individual routes.

    This is done by finding shortest path in a DAG formed by the chomosome
    sequence. The goal of this class is to be able to return how the tour
    should be split by each vehicle showing which vehicle should be
    responsible for which chomosome positions in the tour.
    """

    def split_route(
        self,
        chrm: List[int],
        problem_data: CVRPProblemData,
    ):
        last_pos, vec_used, dist_dict, path_builder = self.builds_dp_structure(
            chrm, problem_data
        )
        vec_dist_dict = self.build_vehicle_path(
            last_pos,
            vec_used,
            dist_dict,
            path_builder,
        )
        return vec_dist_dict

    def builds_dp_structure(
        self,
        chrm: List[int],
        problem_data: CVRPProblemData,
    ) -> Tuple[
        int,
        int,
        Dict[Tuple[int, int], float],
        Dict[Tuple[int, int], Tuple[int, int]],
    ]:
        """
        Funtion to find the shotest path in a DAG. The DAG is the sequence of
        nodes to be visited (defined by the chromosome), where each arc is the
        cost of moving from the depot through every node the arc passes
        (excluding source and including destination) and back to the depot.

        Parameters
        ----------
        chrm : List[int]
            tour sequence given by the chromosome
        problem_data : CVRPProblemData
            problem data with total vehicles, and distance matrix

        Returns
        -------
        Tuple[
            int,
            int,
            Dict[Tuple[int, int], float],
            Dict[Tuple[int, int], Tuple[int, int]],
        ]
            Tuple containing number of vehicles used, final destination reached
            the dynamic programming routes and the path to rebuild the best
            distance found.

        """
        dist_mtx = problem_data.dist_mtx
        prefix_sum = self.prefix_sums(chrm, dist_mtx)
        path = chrm.copy()
        path.insert(0, 0)
        vec_cap = 3
        dist_dict = {}
        dist_dict[(0, 0)] = 0
        path_builder = {}
        vec_used = 0

        for k in range(1, vec_cap + 1):
            for pos, node in enumerate(path[1:], start=1):
                best_cost = math.inf
                for prev_pos, _ in enumerate(path[:pos]):
                    best_so_far = dist_dict.get((prev_pos, k - 1), math.inf)
                    start_of_next_route_pos = prev_pos + 1
                    next_route_path = (
                        prefix_sum[pos] - prefix_sum[start_of_next_route_pos]
                    )
                    next_path_start_node = path[start_of_next_route_pos]
                    depot_to_next_start = dist_mtx[0][next_path_start_node]
                    next_path_end = dist_mtx[node][0]
                    route_cost = (
                        best_so_far
                        + depot_to_next_start
                        + next_route_path
                        + next_path_end
                    )
                    if route_cost < best_cost:
                        source = (prev_pos, k - 1)
                    best_cost = min(best_cost, route_cost)
                if best_cost != math.inf:
                    dist_dict[(pos, k)] = best_cost
                    path_builder[(pos, k)] = source
                    vec_used = k
                    last_pos = pos

        return last_pos, vec_used, dist_dict, path_builder

    @staticmethod
    def prefix_sums(chrm: List[int], dist_mtx: np.ndarray) -> List[float]:
        """
        Pre calculates the distances between nodes in a sequence in O(n).
        Is used as auxiliar for DP algorithm for finding shortest path

        Parameters
        ----------
        chrm : List[int]
            encoded path
        dist_mtx : np.ndarray
            distance between nodes

        Returns
        -------
        List[float]
            list with cumulative distances from node 0
        """
        prefix_sums = [0] * len(chrm)
        for i in range(1, len(chrm)):
            prefix_sums[i] = (
                prefix_sums[i - 1] + dist_mtx[chrm[i - 1]][chrm[i]]
            )

        return prefix_sums

    @staticmethod
    def build_vehicle_path(
        last_pos: int,
        vec_used: int,
        dist_dict: Dict[Tuple[int, int], float],
        path_builder: Dict[Tuple[int, int], Tuple[int, int]],
    ) -> Dict[int, float]:
        """
        From the shortest path output, rebuild the path segregating the
        route size for every specific vehicle.

        Parameters
        ----------
        last_pos : int
            last position reached in the tour
        vec_used : int
            number of vehicles used
        dist_dict : Dict[Tuple[int, int], float]
            dictionary with dynamic programming distances
        path_builder : Dict[Tuple[int, int], Tuple[int, int]]
            maps a node with the one previously visited to reach
            the given distance.

        Returns
        -------
        Dict[int, float]
            dictionary with vehicle number and the path lenght it executed.
        """
        vec_dist_dict = {}
        DAG_path = []

        current_loc = (last_pos, vec_used)
        while vec_used > 0:
            heapq.heappush(DAG_path, current_loc)
            current_loc = path_builder[current_loc]
            vec_used -= 1

        prev_path = 0
        while DAG_path:
            pos, vec = heapq.heappop(DAG_path)
            dist = dist_dict[(pos, vec)]
            vec_dist_dict[vec] = dist - prev_path
            prev_path = dist

        return vec_dist_dict
