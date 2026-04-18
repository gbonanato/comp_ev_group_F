import random
from abc import ABC, abstractmethod
from typing import List


class IndividualInitializer(ABC):
    @abstractmethod
    def generate_chrm():
        pass


class RandomPermInitilizer(IndividualInitializer):
    @staticmethod
    def generate_chrm(chrm_size: int) -> List[int]:
        return random.sample(range(chrm_size), chrm_size)
