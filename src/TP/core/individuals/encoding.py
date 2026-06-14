from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np


class Encoder(ABC):
    @staticmethod
    @abstractmethod
    def validate(chrm: List[Any]) -> List[Any]:
        pass

    @abstractmethod
    def decode(chrm: List[Any]) -> Any:
        pass


class PermutationEncoder(Encoder):
    @staticmethod
    def validate(chrm: List[int]) -> List[int]:
        if len(chrm) != len(set(chrm)):
            print(chrm)
            raise ValueError('Chromossome does not contains unique values.')
        return chrm

    @staticmethod
    def decode(chrm: Union[List[int], List[float]]) -> List[int]:
        has_float = any(isinstance(x, float) for x in chrm)
        if has_float:
            keys = np.array(chrm)
            indices = np.argsort(keys)
            return [int(x) for x in indices]
        else:
            return [int(x) for x in chrm]


class BinaryEncoder(Encoder):
    @staticmethod
    def validate(chrm: List[bool]) -> List[bool]:
        if not all(gene in {0, 1} for gene in chrm):
            raise ValueError('Binary chromosome contains invalid values')
        return chrm

    @staticmethod
    def decode(chrm: List[int]) -> int:
        return int(''.join(map(str, chrm)), 2)
