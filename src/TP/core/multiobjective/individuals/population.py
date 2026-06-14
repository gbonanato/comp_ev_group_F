from abc import ABC
from typing import List, Optional

from pydantic.dataclasses import dataclass

from TP.core.multiobjective.individuals.representation import MOIndividual


# Em teoria parent_selector, recombinator, survivor_selector
# estão atrelados a população, mas não precisam persistir...
# Eles podem ser criados quando utilizados. Guardar pode
# gerar erros por propriedades salvas em iterações anteriores
@dataclass
class MOPopulation(ABC):
    ind_list: List[MOIndividual]
    _select_prob_cache: Optional[List[float]] = None

    def __iter__(self):
        return iter(self.ind_list)

    @property
    def size(self):
        return len(self.ind_list)

    # def get_individuals_fitness(self):
    #     return [i.fitness for i in self.ind_list]
