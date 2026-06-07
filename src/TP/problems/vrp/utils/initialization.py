import random

from pydantic.dataclasses import dataclass

from TP.core.utils.initialization import IndividualInitializer


@dataclass
class VRPInitilizer(IndividualInitializer):
    @staticmethod
    def generate_chrm(chrm_size):
        chrm = random.sample(range(1, chrm_size), (chrm_size - 1))
        return chrm
