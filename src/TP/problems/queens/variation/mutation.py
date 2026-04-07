import random
from itertools import combinations
from typing import List

from TP.core.variation.mutation import MutOperator
from TP.problems.queens.individuals.representation import QueensIndividual


class QueenSwapMutation(MutOperator):
    @staticmethod
    def execute(
        individual: QueensIndividual,
    ) -> List:
        positions = list(range(len(individual.chrm)))
        possible_swaps = combinations(positions, 2)
        if individual.unfit_pos:
            exclude_set = set(individual.unfit_pos)
            incompatible_pos = [
                pos for pos_tuple in exclude_set for pos in pos_tuple
            ]
            significant_swaps = [
                swap
                for pos in incompatible_pos
                for swap in possible_swaps
                if swap not in exclude_set and pos in swap
            ]
            pos_1, pos_2 = random.choice(significant_swaps)
            val_1 = individual.chrm[
                pos_1
            ]  # Saves val from pos_1 before overwriting
            individual.chrm[pos_1] = individual.chrm[pos_2]
            individual.chrm[pos_2] = val_1

            return QueensIndividual(  # Ajustar (chamar um factory)
                individual.chrm, individual.fitness_calculator
            )

        else:
            # pos_1, pos_2 = random.choice(significant_swaps)
            # val_1 = individual.chrm[
            #     pos_1
            # ]  # Saves val from pos_1 before overwriting
            # individual.chrm[pos_1] = individual.chrm[pos_2]
            # individual.chrm[pos_2] = val_1
            # individual.invalidate_fitness()
            # individual.invalidate_unfit_pos()
            return individual
