import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import permutations
from typing import List, Optional

from TP.core.fitness import FitnessCalculator
from TP.core.individuals.representation import Individual
from TP.problems.queens.individuals.representation import QueensIndividual


@dataclass
class RecombOperator(ABC):
    @property
    @abstractmethod
    def n_offsprings(self):
        pass

    @property
    @abstractmethod
    def n_parents(self):
        pass

    @abstractmethod
    def recombine(
        parents_list: List[Individual],
        fitness_calculator: FitnessCalculator,
        n_offsprings: Optional[int],
        p_c: float = 0.7,
    ) -> List[Individual]:
        pass


class PMX(RecombOperator):
    @property
    def n_offsprings(self):
        return 2

    @property
    def n_parents(self):
        return 2

    def recombine(
        self,
        parents_list: List[Individual],
        fitness_calculator: FitnessCalculator,
        p_c: float = 0.7,
    ) -> List[Individual]:
        if len(parents_list) != self.n_parents:
            raise Exception('Only two parents should be used on PMX')

        if random.uniform(0, 1) <= p_c:
            size = len(parents_list[0].chrm)
            lower_cut, upper_cut = sorted(random.sample(range(size), 2))

            offsprings_list = []
            parents_permutation = permutations(parents_list)
            for permutation in parents_permutation:
                chrm = [None] * size
                ref_parent = permutation[0]
                non_ref_parent = permutation[1]
                chrm[lower_cut:upper_cut] = ref_parent.chrm[
                    lower_cut:upper_cut
                ]
                non_ref_parent_mapping = {
                    val: pos for pos, val in enumerate(non_ref_parent.chrm)
                }
                offspring_set = set(chrm)
                for pos, val in enumerate(non_ref_parent.chrm):
                    if val in offspring_set:
                        continue

                    else:
                        while chrm[pos] is not None:
                            map_key = chrm[pos]
                            pos = non_ref_parent_mapping[map_key]
                        chrm[pos] = val
                        offspring_set.add(val)
                offsprings_list.append(
                    QueensIndividual(chrm, fitness_calculator)
                )  # Importante factory para não termos que escolher que tipo de indivíduo é!!

        else:
            offsprings_list = parents_list

        return offsprings_list
