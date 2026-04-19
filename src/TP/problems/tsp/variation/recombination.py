from typing import Any, Dict, List, Set

from tsplib95.models import StandardProblem

from TP.core.individuals.representation import Individual
from TP.core.variation.recombination import RecombOperator


# From: https://www.sciencedirect.com/org/science/article/pii/S1546221824002674
class SCX(RecombOperator):
    @property
    def n_offsprings(self):
        return 1

    @property
    def n_parents(self):
        return 2

    def recombine(
        self,
        parents_list: List[Individual],
        problem_instance: StandardProblem,
        p_c: float = 0.7,
    ) -> List[List[Any]]:
        if len(parents_list) != self.n_parents:
            raise Exception('Only two parents should be used on PMX')

        present_node = 1  # It's our fixed starting point for TSP
        parent_0 = parents_list[0]
        parent_1 = parents_list[1]
        parent_size = len(parent_0.chrm)
        unvisited = set(parent_0.chrm)
        p0_adjacency = {
            parent_0.chrm[i]: parent_0.chrm[i + 1]
            for i in range(parent_size - 1)
        }
        p0_adjacency[parent_0.chrm[parent_size - 1]] = 1
        p1_adjacency = {
            parent_1.chrm[i]: parent_1.chrm[i + 1]
            for i in range(parent_size - 1)
        }
        p1_adjacency[parent_1.chrm[parent_size - 1]] = 1
        child_chrm = []

        while unvisited:
            next_city_candidate_list = []
            p0_candidate = self.next_candidate(
                parent_0,
                p0_adjacency,
                present_node,
                unvisited,
            )

            p1_candidate = self.next_candidate(
                parent_1,
                p1_adjacency,
                present_node,
                unvisited,
            )

            next_city_candidate_list.append(p0_candidate)
            next_city_candidate_list.append(p1_candidate)

            next_city = min(
                next_city_candidate_list,
                key=lambda next: problem_instance.get_weight(
                    present_node, next
                ),
            )

            child_chrm.append(next_city)
            unvisited.remove(next_city)
            p0_candidate = p0_adjacency[next_city]
            p1_candidate = p1_adjacency[next_city]
            present_node = next_city
        return [child_chrm]

    @staticmethod
    def next_candidate(
        parent: Individual,
        adjacency: Dict[int, int],
        present_node: int,
        unvisited: Set,
    ):

        if present_node == 1:
            return parent.chrm[0]
        c = adjacency[present_node]
        if c in unvisited:
            return c
        for city in parent.chrm:
            if city in unvisited:
                return city
