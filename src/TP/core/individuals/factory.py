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

from typing import Any, List

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from TP.core.fitness import FitnessCalculator
from TP.core.individuals.encoding import Encoder
from TP.core.individuals.representation import Individual


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class IndividualFactory:
    fitness_calculator: FitnessCalculator
    encoder: Encoder

    def create(self, chrm: List[Any]) -> Individual:
        return Individual(
            chrm=chrm,
            encoder=self.encoder,
            fitness_calculator=self.fitness_calculator,
        )
