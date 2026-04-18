from typing import List

from TP.core.individuals.encoding import PermutationEncoder


class TSPEncoder(PermutationEncoder):
    @staticmethod
    def decode(chrm: List[int]) -> List[int]:
        if chrm[0] != 1 and chrm[-1] != 1:
            chrm.insert(0, 1)  # Ensures cicle starts
            chrm.append(1)  # and ends on 1
        return chrm
