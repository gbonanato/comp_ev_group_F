from TP.core.fitness import FitnessCalculator


class QueensBoardFitness(FitnessCalculator):
    def calc_fitness(unfit_positions, penalty=-1) -> float:
        fitness = len(unfit_positions) * penalty
        inv_fitness = 1 / abs(fitness)
        return inv_fitness
