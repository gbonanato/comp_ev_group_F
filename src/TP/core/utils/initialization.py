import random
from abc import ABC, abstractmethod


class IndividualInitializer(ABC):
    @abstractmethod
    def generate_chrm():
        pass


@dataclass
class RandomPermInitilizer(IndividualInitializer):
    @staticmethod
    def generate_chrm(chrm_size):
        return random.sample(range(chrm_size), chrm_size)
