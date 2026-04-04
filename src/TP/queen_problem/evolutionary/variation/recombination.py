import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import permutations


@dataclass
class RecombOperator(ABC):
    n_offsprings: int
    n_parents: int

    @property
    @abstractmethod
    def n_offsprings(self):
        pass

    @property
    @abstractmethod
    def n_parents(self):
        pass

    @abstractmethod
    def recombine():
        pass


class PMX(RecombOperator):
    @property
    def n_offsprings(self):
        return 2

    @property
    def n_parents(self):
        return 2

    def recombine(self, parents_list):
        if len(parents_list) != self.n_parents:
            raise Exception('Only two parents should be used on PMX')

        size = len(parents_list[0])
        lower_cut, upper_cut = sorted(random.sample(range(size), 2))

        offsprings_list = []
        parents_permutation = permutations(parents_list)
        for permutation in parents_permutation:
            offspring = [None] * size
            ref_parent = permutation[0]
            non_ref_parent = permutation[1]
            offspring[lower_cut:upper_cut] = ref_parent[lower_cut:upper_cut]
            non_ref_parent_mapping = {
                val: pos for pos, val in enumerate(non_ref_parent)
            }
            offspring_set = set(offspring)
            for pos, val in enumerate(non_ref_parent):
                if val in offspring_set:
                    continue

                else:
                    while offspring[pos] is not None:
                        map_key = offspring[pos]
                        pos = non_ref_parent_mapping[map_key]
                    offspring[pos] = val
                    offspring_set.add(val)
            offsprings_list.append(offspring)

        return offsprings_list
