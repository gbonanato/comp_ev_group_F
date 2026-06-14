from abc import ABC, abstractmethod
from typing import Any, List


class FitnessCalculator(ABC):
    @abstractmethod
    def calc_fitness(self, chmr: List[Any]) -> float:
        pass
