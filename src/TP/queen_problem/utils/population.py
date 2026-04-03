
from abc import ABC
from typing import List

from pydantic.dataclasses import dataclass

from TP.queen_problem.utils.representation import Individual


@dataclass
class Population(ABC):
    ind_list: List[Individual]

    @property
    def pop_size(self):
        return len(self.ind_list)
