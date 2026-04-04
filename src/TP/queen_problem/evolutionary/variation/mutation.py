import random
from abc import ABC, abstractmethod
from ast import Tuple
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional


@dataclass
class MutOperator(ABC):
    @abstractmethod
    def execute():
        pass


class QueenSwapMutation(MutOperator):
    def execute(
        chrm: List[int],
        incompatible_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> List:
        positions = list(range(len(chrm) + 1))
        possible_swaps = combinations(positions)
        if incompatible_pairs:
            exclude_set = set(incompatible_pairs)
            incompatible_pos = [
                pos for pos_tuple in exclude_set for pos in pos_tuple
            ]
            significant_swaps = [
                swap
                for pos in incompatible_pos
                for swap in possible_swaps
                if swap not in exclude_set and pos in swap
            ]
            pos_1, pos_2 = random.choice(significant_swaps)
            val_1 = chrm[pos_1]  # Saves val from pos_1 before overwriting
            chrm[pos_1] = chrm[pos_2]
            chrm[pos_2] = val_1
            return chrm

        else:
            pos_1, pos_2 = random.choice(significant_swaps)
            val_1 = chrm[pos_1]  # Saves val from pos_1 before overwriting
            chrm[pos_1] = chrm[pos_2]
            chrm[pos_2] = val_1
            return chrm
