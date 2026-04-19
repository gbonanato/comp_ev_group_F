import random
from typing import List

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from tsplib95.models import StandardProblem

from TP.core.individuals.representation import Individual
from TP.core.variation.mutation import MutOperator


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CrossAwareRSM(MutOperator):  # Reverse Sequence Mutation
    problem_instance: StandardProblem
    p_random: float = Field(default=0.0, ge=0, le=1)

    def execute(
        self,
        individual: Individual,
    ) -> Individual:
        """
        The Reverse Sequence Mutation (RSM) chooses two randomic positions
        i and j on the chromossome, such that i < j, and inverts the
        order of the information on this section. It is good for structures
        the need to preserve some sense of adjacency, since this operator
        only changes two adjacencies (for j and for i), while preserving the
        overall adjacency.

        Parameters
        ----------
        individual : Individual
            individual to be mutated

        Returns
        -------
        Individual
            Mutated individual
        """
        cross_edges = self.find_crossing(individual.chrm)
        if random.random() < self.p_random:
            i = random.choice(list(range(len(individual.chrm) - 1)))
            j = random.choice(list(range(i + 1, len(individual.chrm) + 1)))
        elif cross_edges:
            i = individual.chrm.index(cross_edges[0][1])
            j = individual.chrm.index(cross_edges[1][0])
        else:
            return individual.chrm

        chrm_section = individual.chrm[i : j + 1].copy()
        chrm_section.reverse()
        new_chromosome = individual.chrm.copy()
        new_chromosome[i : j + 1] = chrm_section

        return new_chromosome

    @staticmethod
    def tour_edges(tour: list[int]):
        edges = []
        prev = 1
        for city in tour:
            edges.append((prev, city))
            prev = city
        edges.append((tour[-1], 1))
        return edges

    def find_crossing(
        self,
        tour: List[int],
    ):
        edges = self.tour_edges(tour)

        for i in range(len(edges) - 3):
            a, b = edges[i]
            for j in range(i + 2, len(edges)):
                c, d = edges[j]

                if self.segments_intersect(
                    self.problem_instance.node_coords[a],
                    self.problem_instance.node_coords[b],
                    self.problem_instance.node_coords[c],
                    self.problem_instance.node_coords[d],
                ):
                    return (edges[i], edges[j])  # indices of crossing edges

        return None

    @staticmethod
    def check_orientation(
        a: List[float],
        b: List[float],
        c: List[float],
    ) -> float:
        """
        EXPLAIN

        Parameters
        ----------
        a : List[float]
            _description_
        b : List[float]
            _description_
        c : List[float]
            _description_

        Returns
        -------
        float
            _description_
        """
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def segments_intersect(
        self,
        a: List[float],
        b: List[float],
        c: List[float],
        d: List[float],
    ):
        o1 = self.check_orientation(a, b, c)
        o2 = self.check_orientation(a, b, d)
        o3 = self.check_orientation(c, d, a)
        o4 = self.check_orientation(c, d, b)

        return o1 * o2 < 0 and o3 * o4 < 0
