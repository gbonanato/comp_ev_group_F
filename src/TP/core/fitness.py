from abc import ABC, abstractmethod


class FitnessCalculator(ABC):
    @abstractmethod
    def calc_fitness():
        pass
