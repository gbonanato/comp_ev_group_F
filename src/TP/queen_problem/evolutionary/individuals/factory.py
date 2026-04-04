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

from pydantic.dataclasses import dataclass

from TP.queen_problem.evolutionary.fitness import QueensBoardFitness
from TP.queen_problem.evolutionary.individuals.representation import (
    Individual,
    PermIndividual,
)
from TP.queen_problem.evolutionary.variation.mutation import QueenSwapMutation


class IndividualFactory(ABC):
    @abstractmethod
    def create(self, chrm: List[Any]) -> Individual:
        pass


@dataclass
class PermIndividualFactory(IndividualFactory):
    fitness_calculator: QueensBoardFitness
    mutation_operator: QueenSwapMutation
    p_m: float = 0.1

    def create(self, chrm: List[int]) -> PermIndividual:

        return PermIndividual(
            chrm=chrm,
            fitness_calculator=self.fitness_calculator,
            mutation_operator=self.mutation_operator,
            p_m=self.p_m,
        )
