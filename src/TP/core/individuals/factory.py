"""
This module provides decoupling from individual creation and recombination
strategy.

Recombination operators will create genotipical information, but other
individual relevant informations, such as mutation rate, fitness calculator
and mutation operator can be easily tunned if needed after recombination.

Classes:
    IndividualFactory: abstract implementation of how to re-create a individual

Functions:
    create: creates individual.
"""

from abc import ABC, abstractmethod
from typing import Any, List

from TP.core.individuals.representation import Individual


class IndividualFactory(ABC):
    @abstractmethod
    def create(self, chrm: List[Any]) -> Individual:
        pass
