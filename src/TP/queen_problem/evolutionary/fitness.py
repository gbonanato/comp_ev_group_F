from abc import ABC, abstractmethod


class FitnessCalculator(ABC):
    @abstractmethod
    def calc_fitness():
        pass


class QueensBoardFitness(FitnessCalculator):
    def calc_fitness(unfit_positions, penalty=-1) -> float:
        fitness = len(unfit_positions) * penalty
        inv_fitness = 1 / abs(fitness)
        return inv_fitness
