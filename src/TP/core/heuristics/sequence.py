from typing import List, Union

import numpy as np

from TP.core.individuals.representation import Individual
from TP.core.multiobjective.individuals.representation import MOIndividual

# CrossAwareRSM is in fact 2-opt operation. On 2-opt we must check
# for every pair if changing the supposed crossed edges the overal
# path lenght is improved or not. If it improves, we perform the inversion.


def run_2opt(
    indiv: Union[MOIndividual, Individual], dist_mtx: np.ndarray
) -> List[int]:
    cross_edges = find_2_opt(indiv.chrm, dist_mtx)
    if cross_edges:
        i = indiv.chrm.index(cross_edges[0][1])
        j = indiv.chrm.index(cross_edges[1][0])
    else:
        return indiv.chrm

    return creates_tour_inverted_chrm(indiv, i, j)


def tour_edges(tour: list[int]):
    edges = []
    prev = 0
    for city in tour:
        edges.append((prev, city))
        prev = city
    edges.append((tour[-1], 0))
    return edges


def find_2_opt(tour: List[int], dist_mtx):
    edges = tour_edges(tour)

    for i in range(len(edges) - 3):
        a, b = edges[i]
        if a == 0 or b == 0:
            continue

        for j in range(i + 2, len(edges) - 1):
            c, d = edges[j]
            if c == 0 or d == 0:
                continue

            if check_inversion_improvement(a, b, c, d, dist_mtx):
                return (edges[i], edges[j])  # indexes of crossing edges

    return None


def check_inversion_improvement(
    a: int, b: int, c: int, d: int, dist_mtx: np.ndarray
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
    inversion_cost = dist_mtx[a, c] + dist_mtx[b, d]
    edges_cost = dist_mtx[a, b] + dist_mtx[c, d]
    if inversion_cost < edges_cost:
        return True
    return False


def creates_tour_inverted_chrm(
    individual: Union[Individual, MOIndividual],
    i: int,
    j: int,
) -> List[int]:
    chrm_section = individual.chrm[i : j + 1].copy()
    chrm_section.reverse()
    new_chromosome = individual.chrm.copy()
    new_chromosome[i : j + 1] = chrm_section

    return new_chromosome
