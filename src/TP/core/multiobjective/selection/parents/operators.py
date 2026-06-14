import random
from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic.dataclasses import dataclass

from TP.core.multiobjective.individuals.population import MOPopulation
from TP.core.multiobjective.individuals.representation import MOIndividual
from TP.core.state import EAState


@dataclass
class MOParentSelector(ABC):
    @abstractmethod
    def select_parents(
        self, pop: MOPopulation, num_parents: int, state: Optional[EAState]
    ) -> List[MOIndividual]:
        pass


# ANtes de executar precisa atualiza o state.
@dataclass
class CrowdTournament(MOParentSelector):
    tournament_size: int = 2

    def select_parents(
        self, pop: MOPopulation, num_parents: int, state: Optional[EAState]
    ) -> List[MOIndividual]:
        if state is None:
            raise ValueError('Crowd Tournament needs a state definition.')
        elif state.rank is None:
            raise ValueError('Crowd Tournament needs a rank definition.')
        elif state.crowding is None:
            raise ValueError(
                'Crowd Tournament needs a crowding distance definition.'
            )
        else:
            parents_list = []
            for _ in range(num_parents):
                parents_candidates = random.sample(
                    pop.ind_list,
                    k=self.tournament_size,
                )
                best_rank = min(
                    state.rank[indiv] for indiv in parents_candidates
                )
                highest_rank_indiv = [
                    indiv
                    for indiv in parents_candidates
                    if state.rank[indiv] == best_rank
                ]
                if len(highest_rank_indiv) > 1:
                    selected_parent = max(
                        highest_rank_indiv,
                        key=lambda indiv: state.crowding[indiv],
                    )
                else:
                    selected_parent = highest_rank_indiv[0]

                parents_list.append(selected_parent)

        return parents_list
