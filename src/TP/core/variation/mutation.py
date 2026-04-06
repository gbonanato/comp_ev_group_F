from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MutOperator(ABC):
    @abstractmethod
    def execute():
        pass
