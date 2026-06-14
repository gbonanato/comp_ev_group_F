# TP/core/logging/base.py
import logging
from typing import Optional

from pydantic.dataclasses import dataclass

from TP.core.individuals.population import Population
from TP.core.state import EAState


@dataclass
class EALogger:
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
        if isinstance(pop, Population):
            best = max(ind.fitness for ind in pop.ind_list)
            mean = sum(ind.fitness for ind in pop.ind_list) / pop.size

            self.logger.info(
                'Generation %d | best=%.5f | mean=%.5f | Pop Size=%d',
                state.generation,
                best,
                mean,
                pop.size,
            )
        elif state.fronts:
            self.logger.info(
                'Generation %d | Best front size=%.0f | Pop Size=%d',
                state.generation,
                len(state.fronts[0]),
                pop.size,
            )

    def on_end(self, state: EAState):
        if isinstance(state.population, Population):
            best = max(ind.fitness for ind in state.population.ind_list)
            self.logger.info(
                'EA finished | generations=%d | best_fitness=%.5f | feasible=%s',
                state.generation,
                best,
                state.feasibility,
            )
        elif state.fronts:
            self.logger.info(
                'EA finished | generations=%d | best front size=%.5f | feasible=%s',
                state.generation,
                len(state.fronts[0]),
                state.feasibility,
            )
