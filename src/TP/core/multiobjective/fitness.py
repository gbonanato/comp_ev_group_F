from abc import ABC, abstractmethod


class MOFitnessCalculator(ABC):
    @abstractmethod
    def calc_fitness():
        pass
