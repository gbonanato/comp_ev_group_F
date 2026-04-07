from typing import List

from TP.core.individuals.population import Population
from TP.problems.queens.individuals.representation import QueensIndividual


class QueensPopulation(Population):
    ind_list: List[QueensIndividual]
