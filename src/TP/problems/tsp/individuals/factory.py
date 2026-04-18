"""
This module provides decoupling from individual creation and recombination
strategy.

Recombination operators will create genotipical information, but other
individual relevant informations, such as mutation rate, fitness calculator
and mutation operator can be easily tunned if needed after recombination
using individual factory.

Classes:
    IndividualFactory: abstract implementation of how to re-create a individual

Functions:
    create: creates individual.
"""

from typing import List

from pydantic.dataclasses import dataclass

from TP.core.individuals.factory import IndividualFactory
from TP.problems.queens.fitness import QueensBoardFitness
from TP.problems.queens.individuals.representation import QueensIndividual
from TP.problems.queens.variation.mutation import QueenSwapMutation


@dataclass
class PermIndividualFactory(IndividualFactory):
    fitness_calculator: QueensBoardFitness
    mutation_operator: QueenSwapMutation
    p_m: float = 0.1

    def create(self, chrm: List[int]) -> QueensIndividual:

        return QueensIndividual(
            chrm=chrm,
            fitness_calculator=self.fitness_calculator,
            mutation_operator=self.mutation_operator,
            p_m=self.p_m,
        )
