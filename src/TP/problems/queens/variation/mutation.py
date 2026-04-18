import random
from itertools import combinations
from typing import List

from TP.core.individuals.representation import Individual
from TP.core.variation.mutation import MutOperator
from TP.problems.queens.individuals.representation import QueensIndividual


class QueenSwapMutation(MutOperator):
    @staticmethod
    def execute(
        individual: QueensIndividual,
    ) -> Individual:
        positions = list(range(len(individual.chrm)))
        possible_swaps = list(combinations(positions, 2))
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

            if not significant_swaps:
                return individual

            pos_1, pos_2 = random.choice(significant_swaps)
            new_chrm = individual.chrm.copy()
            new_chrm[pos_1], new_chrm[pos_2] = new_chrm[pos_2], new_chrm[pos_1]

            return QueensIndividual(  # Ajustar (chamar um factory)
                new_chrm, individual.fitness_calculator
            )

        else:
            return individual
