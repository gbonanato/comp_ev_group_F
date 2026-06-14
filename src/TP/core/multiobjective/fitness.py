from abc import ABC, abstractmethod
from typing import Any, List, NamedTuple

from pydantic.dataclasses import dataclass


class Objective(NamedTuple):
    name: str
    value: float
    higher_is_better: bool


class ProblemMetric(NamedTuple):
    metric1: Objective
    metric2: Objective


@dataclass
class MOFitnessCalculator(ABC):
    @abstractmethod
    def calc_fitness(self, chrm: List[Any]) -> ProblemMetric:
        pass
