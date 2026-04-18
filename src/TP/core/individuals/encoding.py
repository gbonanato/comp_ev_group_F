from abc import ABC, abstractmethod
from typing import Any, List


class Encoder(ABC):
    @abstractmethod
    def validate(chrm: List[Any]) -> None:
        pass

    @abstractmethod
    def decode(chrm: List[Any]) -> Any:
        pass


class PermutationEncoder(Encoder):
    @staticmethod
    def validate(chrm: List[int]) -> None:
        if len(chrm) != len(set(chrm)):
            print(chrm)
            raise ValueError('Chromossome does not contains unique values.')
        return chrm

    @staticmethod
    def decode(chrm: List[int]) -> List[int]:
        return chrm


class BinaryEncoder(Encoder):
    @staticmethod
    def validate(chrm: List[int]) -> None:
        if not all(gene in {0, 1} for gene in chrm):
            raise ValueError('Binary chromosome contains invalid values')

    @staticmethod
    def decode(chrm: List[int]) -> int:
        return int(''.join(map(str, chrm)), 2)
