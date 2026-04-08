# TP/core/logging/observer.py
from abc import ABC, abstractmethod
from typing import List

from TP.core.state import EAState


class EAObserver(ABC):
    @abstractmethod
    def on_generation_end(self, state: EAState):
        """
        Chamado ao final de cada geração
        """
        pass

    def on_start(self, state: EAState):
        """
        Opcional: chamado no início da execução
        """
        pass

    def on_end(self, state: EAState):
        """
        Opcional: chamado no fim da execução
        """
        pass


class BestFitnessLogger(EAObserver):
    def __init__(self):
        self.generations: List[int] = []
        self.best_fitness: List[float] = []

    def on_generation_end(self, state: EAState):
        population = state.population
        best = max(ind.fitness for ind in population.ind_list)

        self.generations.append(state.generation)
        self.best_fitness.append(best)
