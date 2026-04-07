import random

from pydantic.dataclasses import dataclass

from TP.core.utils.initialization import IndividualInitializer


@dataclass
class RandomPermInitilizer(IndividualInitializer):
    @staticmethod
    def generate_chrm(chrm_size):
        return random.sample(range(chrm_size), chrm_size)
