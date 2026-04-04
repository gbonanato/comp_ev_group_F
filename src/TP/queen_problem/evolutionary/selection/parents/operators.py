"""
This module provides implementations of parent selection operators.

Operators are responsible for attributing selection probabilities to
individuals in a population.

Classes:
    SelectOperator: abstract implementation of what a operator must have.
    OperatorStrategy: classes that holds implementations of operators for
    selection during runtime.

Functions:
    calc_selection_prob: Calculates individuals selection probabilities.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from pydantic.dataclasses import dataclass

from TP.queen_problem.evolutionary.individuals.population import Population


class ParentSelector(ABC):
    @abstractmethod
    def select_parents(
        self,
        pop: Population,
        num_parents: int,
    ) -> List[int]:
        pass


@dataclass
class RouletteStrategy(ParentSelector):
    pop: Population
    _selection_prob: Optional[List[float]] = None

    @property
    def selection_prob(self):
        if self._selection_prob is None:
            self._selection_prob = self._selection_prob(self.pop)
        return self._selection_prob

    def calc_selection_prob(pop: Population) -> List[float]:
        indiv_fitness = [ind.get_fitness() for ind in pop.ind_list]
        indiv_fitness = np.array(indiv_fitness)
        total_fitness = sum(indiv_fitness)
        return indiv_fitness / total_fitness

    def select_parents(
        self,
        num_parents: int,
    ) -> List[int]:
        fitness_list = self.selection_prob
        parents = random.choices(
            range(len(fitness_list)),
            weights=fitness_list,
            k=num_parents,
        )
        return parents
