import random

from pydantic.dataclasses import dataclass

from TP.core.utils.initialization import IndividualInitializer


@dataclass
class RandomPermInitilizer(IndividualInitializer):
    chrm_size: int

    def generate_chrm(self):
        return random.sample(range(self.chrm_size), self.chrm_size)
