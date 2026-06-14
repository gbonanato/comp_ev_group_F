import math
from typing import Dict, List

from TP.core.multiobjective.individuals.representation import MOIndividual


def compute_crowding_distance(
    indiv_list: List[MOIndividual],
) -> Dict[MOIndividual, float]:
    ind_distance_dict = dict.fromkeys(indiv_list, 0.0)
    # Starts individuals crowd distances as zero

    num_of_objectives = len(indiv_list[0].fitness)

    for part_obj in range(num_of_objectives):
        # The iteration assumes partial objectives positions are the same
        # so they can be compared by position 0, 1, 2...
        sorted_indiv_list = sorted(
            indiv_list,
            key=lambda ind: ind.fitness[part_obj].value,
        )
        f_min = sorted_indiv_list[0].fitness[part_obj].value
        f_max = sorted_indiv_list[-1].fitness[part_obj].value

        if f_max == f_min:
            continue

        ind_distance_dict[sorted_indiv_list[0]] = math.inf
        ind_distance_dict[sorted_indiv_list[-1]] = math.inf

        for i in range(1, len(sorted_indiv_list) - 1):
            prev_f_dist = sorted_indiv_list[i - 1].fitness[part_obj].value
            next_f_dist = sorted_indiv_list[i - 1].fitness[part_obj].value
            ind_dist = (next_f_dist - prev_f_dist) / (f_max - f_min)
            ind_distance_dict[sorted_indiv_list[i]] += ind_dist

    return ind_distance_dict
