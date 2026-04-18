import random

from pydantic.dataclasses import dataclass

from TP.core.utils.initialization import IndividualInitializer


@dataclass
class TSPInitilizer(IndividualInitializer):
    @staticmethod
    def generate_chrm(chrm_size):
        chrm = random.sample(range(1, chrm_size), (chrm_size - 1))
        chrm.insert(0, 0)
        chrm.append(0)
        return chrm
