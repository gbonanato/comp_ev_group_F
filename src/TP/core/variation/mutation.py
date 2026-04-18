import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from TP.core.individuals.representation import Individual


@dataclass
class MutOperator(ABC):
    @abstractmethod
    def execute(Individual) -> Individual:
        pass


@dataclass
class RSM(MutOperator):  # Reverse Sequence Mutation
    def execute(
        self,
        individual: Individual,
    ) -> Individual:
        """
        The Reverse Sequence Mutation (RSM) chooses two randomic positions
        i and j on the chromossome, such that i < j, and inverts the
        order of the information on this section. It is good for structures
        the need to preserve some sense of adjacency, since this operator
        only changes two adjacencies (for j and for i), while preserving the
        overall adjacency.

        Parameters
        ----------
        individual : Individual
            individual to be mutated

        Returns
        -------
        Individual
            Mutated individual
        """
        i = random.choice(list(range(len(individual.chrm) - 1)))
        j = random.choice(list(range(i + 1, len(individual.chrm) + 1)))
        chrm_section = individual.chrm[i:j].copy()
        chrm_section.reverse()
        new_chromosome = individual.chrm.copy()
        new_chromosome[i:j] = chrm_section

        return new_chromosome
