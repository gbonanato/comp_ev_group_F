from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic.dataclasses import dataclass

from TP.queen_problem.evolutionary.individuals.factory import (
    IndividualFactory,
)
from TP.queen_problem.evolutionary.individuals.representation import Individual
from TP.queen_problem.evolutionary.selection.parents.operators import (
    ParentSelector,
)
from TP.queen_problem.evolutionary.selection.survivors.operators import (
    SurvivorSelector,
)
from TP.queen_problem.evolutionary.variation.recombination import (
    RecombOperator,
)


@dataclass
class Population(ABC):
    ind_list: List[Individual]
    parent_selector: ParentSelector
    recombinator: RecombOperator
    survivor_selector: SurvivorSelector
    indiv_factory: IndividualFactory

    def __iter__(self):
        return iter(self.ind_list)

    @property
    def pop_size(self):
        return len(self.ind_list)

    def get_individuals_fitness(self):
        return [i.fitness for i in self.ind_list]

    @abstractmethod
    def generate_offsprings(self):
        pass

    @abstractmethod
    def select_next_generation(self):
        pass


class QueensPopulation(Population):
    def generate_offsprings(
        self,
        n_offsprings: int = None,
    ) -> List[Individual]:
        if n_offsprings is None:
            n_offsprings = len(self.ind_list)
        assert n_offsprings % self.recombinator.n_offsprings == 0

        offspings_list = []

        while len(offspings_list) < n_offsprings:
            parents_list = self.parent_selector.select_parents(
                self.ind_list, self.recombinator.n_parents
            )
            children = self.recombinator.recombine(parents_list)

            for child in children:
                offspings_list.append(self.indiv_factory(child))

        return offspings_list

    def select_next_generation(
        self,
        offspings_list: List[Individual],
        n_survivors: Optional[int],
    ):
        if n_survivors is None:
            n_survivors = self.pop_size
        next_gen = self.survivor_selector.select_survivors(
            self.ind_list,
            offspings_list,
            n_survivors,
        )
        return next_gen
