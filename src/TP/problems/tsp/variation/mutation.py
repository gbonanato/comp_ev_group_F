import random
from typing import List

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from TP.core.individuals.representation import Individual
from TP.core.variation.mutation import MutOperator

# CrossAwareRSM is in fact 2-opt operation. On 2-opt we must check
# for every pair if changing the supposed crossed edges the overal
# path lenght is improved or not. If it improves, we perform the inversion.


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CrossAwareRSM(MutOperator):  # Reverse Sequence Mutation REFACTOR
    problem_instance: np.ndarray
    p_random: float = Field(default=0.2, ge=0, le=1)

    def execute(
        self,
        individual: Individual,
    ) -> Individual:
        """
        The Reverse Sequence Mutation (RSM) chooses two randomic positions
        i and j on the chromossome, such that i < j, and inverts the
        order of the information on this section. It is good for structures
        that need to preserve some sense of adjacency, since this operator
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

        cross_edges = self.find_2_opt(individual.chrm)
        if random.random() < self.p_random:
            i = random.choice(list(range(len(individual.chrm) - 1)))
            j = random.choice(list(range(i + 1, len(individual.chrm) + 1)))
        elif cross_edges:
            i = individual.chrm.index(cross_edges[0][1])
            j = individual.chrm.index(cross_edges[1][0])
        else:
            return individual.chrm

        return self.creates_tour_inverted_chrm(individual, i, j)

    @staticmethod
    def tour_edges(tour: list[int]):
        edges = []
        prev = 0
        for city in tour:
            edges.append((prev, city))
            prev = city
        edges.append((tour[-1], 0))
        return edges

    def find_2_opt(
        self,
        tour: List[int],
    ):
        edges = self.tour_edges(tour)

        for i in range(len(edges) - 3):
            a, b = edges[i]
            if a == 0 or b == 0:
                continue

            for j in range(i + 2, len(edges) - 1):
                c, d = edges[j]
                if c == 0 or d == 0:
                    continue

                if self.check_inversion_improvement(a, b, c, d):
                    return (edges[i], edges[j])  # indexes of crossing edges

        return None

    def check_inversion_improvement(
        self,
        a: int,
        b: int,
        c: int,
        d: int,
    ) -> bool:
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
        inversion_cost = (
            self.problem_instance[a, c] + self.problem_instance[b, d]
        )
        edges_cost = self.problem_instance[a, b] + self.problem_instance[c, d]
        if inversion_cost < edges_cost:
            return True
        return False

    @staticmethod
    def creates_tour_inverted_chrm(individual: Individual, i: int, j: int):
        chrm_section = individual.chrm[i : j + 1].copy()
        chrm_section.reverse()
        new_chromosome = individual.chrm.copy()
        new_chromosome[i : j + 1] = chrm_section

        return new_chromosome
