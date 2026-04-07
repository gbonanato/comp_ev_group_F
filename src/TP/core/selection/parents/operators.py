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
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from TP.core.individuals.population import Population
from TP.core.individuals.representation import Individual


class ParentSelector(ABC):
    @abstractmethod
    def select_parents(
        self,
        num_parents: int,
        pop: Population,
    ) -> List[int]:
        pass


@dataclass
class RouletteStrategy(ParentSelector):
    @staticmethod
    def calc_selection_prob(pop: Population) -> List[float]:
        indiv_fitness = [ind.fitness for ind in pop.ind_list]
        indiv_fitness = np.array(indiv_fitness)
        total_fitness = sum(indiv_fitness)
        if total_fitness == 0:
            return [1 / len(indiv_fitness)] * len(indiv_fitness)

        return [f / total_fitness for f in indiv_fitness]

    def select_parents(
        self,
        num_parents: int,
        pop: Population,
    ) -> List[Individual]:
        parents_list = []
        if pop._select_prob_cache is None:
            pop._select_prob_cache = self.calc_selection_prob(pop)
        fitness_list = pop._select_prob_cache
        parents_pos = random.choices(
            range(len(fitness_list)),
            weights=fitness_list,
            k=num_parents,
        )
        for pos in parents_pos:
            parents_list.append(pop.ind_list[pos])
        return parents_list
