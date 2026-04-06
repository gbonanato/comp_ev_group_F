from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic.dataclasses import dataclass

from TP.core.individuals.factory import IndividualFactory
from TP.core.individuals.representation import Individual
from TP.core.selection.parents.operators import ParentSelector
from TP.core.selection.survivors.operators import SurvivorSelector
from TP.core.variation.recombination import RecombOperator


# Em teoria parent_selector, recombinator, survivor_selector
# estão atrelados a população, mas não precisam persistir...
# Eles podem ser criados quando utilizados. Guardar pode
# gerar erros por propriedades salvas em iterações anteriores
@dataclass
class Population(ABC):
    ind_list: List[Individual]
    parent_selector: ParentSelector
    recombinator: RecombOperator
    survivor_selector: SurvivorSelector
    indiv_factory: IndividualFactory
    _select_prob_cache: Optional[List[float]]
    p_c: float

    def __iter__(self):
        return iter(self.ind_list)

    @property
    def size(self):
        return len(self.ind_list)

    def get_individuals_fitness(self):
        return [i.fitness for i in self.ind_list]

    @abstractmethod
    def generate_offsprings(self):
        pass

    @abstractmethod
    def select_next_generation(
        mutated_offsprings: List[Individual],
        n_survivors: Optional[int],
    ):
        pass
