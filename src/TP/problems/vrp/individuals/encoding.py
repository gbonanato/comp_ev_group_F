from typing import List

from TP.core.individuals.encoding import PermutationEncoder


class TSPEncoder(PermutationEncoder):
    @staticmethod
    def decode(chrm: List[int]) -> List[int]:
        if chrm[0] != 0 and chrm[-1] != 0:
            chrm.insert(0, 0)  # Ensures cicle starts
            chrm.append(0)  # and ends on 0
        return chrm
