# TP/core/logging/base.py
import logging
from abc import ABC, abstractmethod
from typing import Optional

from pydantic.dataclasses import dataclass

from TP.core.state import EAState


class EALogger(ABC):
    """Side-effect logger for EA execution."""

    @abstractmethod
    def on_start(self, state: EAState):
        pass

    @abstractmethod
    def on_generation_end(self, state: EAState):
        pass

    @abstractmethod
    def on_end(self, state: EAState):
        pass


@dataclass
class EAProgressLogger(EALogger):
    logger = logging.getLogger(__name__)
    frequecy: Optional[int] = 1

    def on_start(self, state: EAState):
        self.logger.info(
            'EA started | population_size=%d',
            state.population.size,
        )

    def on_generation_end(self, state: EAState):
        if state.generation % self.frequecy != 0:
            return

        pop = state.population
        best = max(ind.fitness for ind in pop.ind_list)
        mean = sum(ind.fitness for ind in pop.ind_list) / pop.size

        self.logger.info(
            'Generation %d | best=%.5f | mean=%.5f',
            state.generation,
            best,
            mean,
        )

    def on_end(self, state: EAState):
        best = max(ind.fitness for ind in state.population.ind_list)
        self.logger.info(
            'EA finished | generations=%d | best_fitness=%.5f | feasible=%s',
            state.generation,
            best,
            state.feasibility,
        )
