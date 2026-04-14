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

        # Converter fitness (-conflicts) → conflicts positivos
        conflicts = [-f for f in indiv_fitness]

        epsilon = 1e-6

        # Evitar caso degenerado (todos iguais)
        if all(c == conflicts[0] for c in conflicts):
            return [1 / len(conflicts)] * len(conflicts)

        # Converter para pesos (menor conflito → maior peso)
        weights = [1 / (c + epsilon) for c in conflicts]

        total = sum(weights)

        return [w / total for w in weights]

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
