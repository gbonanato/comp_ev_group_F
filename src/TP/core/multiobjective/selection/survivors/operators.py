from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic.dataclasses import dataclass

from TP.core.multiobjective.individuals.population import MOPopulation
from TP.core.multiobjective.individuals.representation import MOIndividual
from TP.core.state import EAState


@dataclass
class MOSurvivorSelector(ABC):
    @staticmethod
    @abstractmethod
    def select_survivors(
        parents: List[MOIndividual],
        offsprings: List[MOIndividual],
        n_survivors: Optional[int],
        state: Optional[EAState],
    ) -> MOPopulation:
        pass


@dataclass
class NonDominanceOrdering(MOSurvivorSelector):
    @staticmethod
    def select_survivors(
        parents: List[MOIndividual],
        offsprings: List[MOIndividual],
        n_survivors: Optional[int],
        state: Optional[EAState],
    ) -> MOPopulation:
        if n_survivors is None:
            n_survivors = len(parents)
        if state is None:
            raise ValueError('Crowd Tournament needs a state definition.')
        else:
            new_pop = []
            pop_indiv_count = 0

            front_total = 0
            for front in state.fronts:
                front_total += len(front)

            for front in state.fronts:
                remaining = n_survivors - pop_indiv_count

                if remaining <= 0:
                    break

                if len(front) <= remaining:
                    new_pop.extend(front)
                    pop_indiv_count += len(front)
                else:
                    sorted_front = sorted(
                        front,
                        key=lambda indiv: state.crowding[indiv],
                        reverse=True,
                    )
                    new_pop.extend(sorted_front[:remaining])
                    pop_indiv_count += remaining
                    break

            # safety guard (opcional mas recomendado)
            assert len(new_pop) == n_survivors, (
                f'Expected {n_survivors}, got {len(new_pop)}'
            )
        return MOPopulation(ind_list=new_pop)
