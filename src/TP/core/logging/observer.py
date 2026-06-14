# TP/core/logging/observer.py
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Union

from TP.core.individuals.population import Population
from TP.core.state import EAState


class EAObserver(ABC):
    @abstractmethod
    def on_generation_end(self, state: EAState):
        """
        Chamado ao final de cada geração
        """
        pass

    # @abstractmethod
    # def on_start(self, state: Union[EAState, EAState]):
    #     """
    #     Opcional: chamado no início da execução
    #     """
    #     pass

    # @abstractmethod
    # def on_end(self, state: Union[EAState, EAState]):
    #     """
    #     Opcional: chamado no fim da execução
    #     """
    #     pass


class BestFitnessLogger(EAObserver):
    def __init__(self):
        self.generations: List[int] = []
        self.best_fitness: List[Union[float, List[NamedTuple]]] = []

    def on_generation_end(self, state: EAState):

        if state.fronts is not None:
            best = [
                ind.fitness
                for ind in state.fronts[0]
                if ind.fitness is not None
            ]

        elif isinstance(state.population, Population):
            best = max(ind.fitness for ind in state.population.ind_list)

        else:
            raise ValueError('Best result for generation cannot be None')

        self.generations.append(state.generation)
        self.best_fitness.append(best)
