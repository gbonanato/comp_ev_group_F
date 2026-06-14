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
    ) -> List[Individual]:
        pass


@dataclass
class RouletteStrategy(ParentSelector):
    @staticmethod
    def calc_selection_prob(pop: Population) -> List[float]:
        pop_fitness = [ind.fitness for ind in pop.ind_list]

        total = sum(pop_fitness)

        return [indiv_fitness / total for indiv_fitness in pop_fitness]

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


@dataclass
class RankRouletteStrategy(ParentSelector):
    selection_pressure: float = 1.7  # usually in [1.5, 2.0]

    def calc_selection_prob(self, pop: Population) -> List[float]:
        n = len(pop.ind_list)
        if n == 0:
            raise ValueError('Population is empty')

        # Sort individuals by fitness (ascending)
        sorted_inds = sorted(
            enumerate(pop.ind_list),
            key=lambda x: x[1].fitness,
        )

        # ranks: worst = 1, best = n
        ranks = [0] * n
        for rank, (idx, _) in enumerate(sorted_inds, start=1):
            ranks[idx] = rank

        s = self.selection_pressure

        # Baker's linear ranking probabilities
        probs = [
            (2 - s) / n + (2 * rank * (s - 1)) / (n * (n - 1))
            for rank in ranks
        ]

        return probs

    def select_parents(
        self,
        num_parents: int,
        pop: Population,
    ) -> List[Individual]:

        if pop._select_prob_cache is None:
            pop._select_prob_cache = self.calc_selection_prob(pop)

        fitness_list = pop._select_prob_cache

        chosen_indices = random.choices(
            range(len(fitness_list)),
            weights=fitness_list,
            k=num_parents,
        )

        return [pop.ind_list[i] for i in chosen_indices]


@dataclass
class TournamentStrategy(ParentSelector):
    tournament_size: int = 2

    @staticmethod
    def uniform_selection_prob(pop: Population) -> List[float]:
        n = len(pop.ind_list)
        return [1 / n] * n

    def select_parents(
        self,
        num_parents: int,
        pop: Population,
        prob: str = 'uniform',
    ) -> List[Individual]:
        parents_list = []
        if pop._select_prob_cache is None:
            if prob == 'uniform':
                pop._select_prob_cache = self.uniform_selection_prob(pop)
        select_prob = pop._select_prob_cache
        for _ in range(num_parents):
            parents_pos = random.sample(
                range(len(pop.ind_list)),
                # weights=select_prob,
                k=self.tournament_size,
            )
            candidates = [pop.ind_list[pos] for pos in parents_pos]
            winner = max(candidates, key=lambda c: c.fitness)
            parents_list.append(winner)

        return parents_list


# TODO:
# Ajustar: se for tourneio com roleta, tem que usar random.choice com os pesos.
# Isso vai permitir um sample com replacement. Porém podemos querer fazer sem
# replacement, que é o caso de usar random.sample.
# Temos que definir estratégias para esses use cases.
