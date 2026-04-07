from abc import ABC, abstractmethod
from dataclasses import dataclass

from TP.core.individuals.representation import Individual


@dataclass
class MutOperator(ABC):
    @abstractmethod
    def execute(Individual) -> Individual:
        pass
