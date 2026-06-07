from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from TP.core.fitness import FitnessCalculator
from TP.core.individuals.encoding import Encoder
from TP.core.individuals.factory import IndividualFactory
from TP.core.individuals.population import Population
from TP.core.individuals.representation import Individual
from TP.core.logging.observer import EAObserver
from TP.core.logging.progress import EALogger
from TP.core.selection.parents.operators import ParentSelector
from TP.core.selection.survivors.operators import SurvivorSelector
from TP.core.state import EAState
from TP.core.utils.initialization import IndividualInitializer
from TP.core.variation.mutation import MutOperator
from TP.core.variation.recombination import RecombOperator


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class NSGAIIOrchestratorTemplate(ABC):
    pop_size: int

    encoder: Encoder
    parent_selector: ParentSelector
    recombinator: RecombOperator
    survivor_selector: SurvivorSelector

    mutation_operator: MutOperator

    ind_initializer: IndividualInitializer
    observers: List[EAObserver]
    loggers: List[EALogger]

    # defaults
    fitness_calculator: Optional[FitnessCalculator] = None
    individual_factory: Optional[IndividualFactory] = None
    max_generations: Optional[int] = 1000
    n_parents: Optional[int] = 2
    n_offsprings: Optional[int] = 2
    p_m: float = 0.1
    p_c: float = 0.7

    def __post_init__(self):
        self._configure_fitness()
        self._configure_factory()

    @abstractmethod
    def _configure_fitness(self):
        """Subclass defines fitness semantics"""

    def _configure_factory(self):
        if self.individual_factory is None:
            if self.fitness_calculator is None:
                raise RuntimeError('Fitness must be configured first')

            self.individual_factory = IndividualFactory(
                fitness_calculator=self.fitness_calculator,
                encoder=self.encoder,
            )

    @abstractmethod
    def generate_individual(self) -> Individual:
        """
        Logic to create initial population

        Returns
        -------
        Population
            Initial problem population.
        """
        pass
