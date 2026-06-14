from typing import Dict, List, Tuple

from TP.core.multiobjective.individuals.population import MOPopulation
from TP.core.multiobjective.individuals.representation import MOIndividual


def fast_non_dominated_sort(
    population: MOPopulation,
) -> List[List[MOIndividual]]:

    rank_list, dominated_by_ind, dominates_ind = start_ranking(
        population.ind_list
    )
    all_rank_list = [rank_list]
    rank_count = 0
    while all_rank_list[rank_count]:
        rank_queue = []
        for indiv in all_rank_list[rank_count]:
            for q in dominated_by_ind[indiv]:
                dominates_ind[q] -= 1
                if dominates_ind[q] == 0:
                    if q not in rank_queue:
                        rank_queue.append(q)

        if rank_queue:
            rank_count += 1
            all_rank_list.append(rank_queue)
        else:
            break

    total = sum(len(front) for front in all_rank_list)
    assert total == len(population.ind_list), (
        f'Inconsistent fronts: {total} != {len(population.ind_list)}'
    )

    return all_rank_list


def check_ind_dominance(ind: MOIndividual, other_ind: MOIndividual):
    better_in_at_least_one = False
    ind_fitness = ind.fitness
    other_fitness = other_ind.fitness
    if ind_fitness is not None and other_fitness is not None:
        if len(ind_fitness) == 1 or len(other_fitness) == 1:
            raise ValueError(
                'Individuals are mono objective, and should not use'
                'dominance based selection'
            )

        for part_obj_ind, part_obj_other_ind in zip(
            ind_fitness, other_fitness, strict=True
        ):
            if part_obj_ind.name != part_obj_other_ind.name:
                raise KeyError(
                    f"Can't compare partial objective {part_obj_ind.name},"
                    'with {part_obj_other_ind.name}.'
                    'Check fitness object creation'
                )
            if part_obj_ind.higher_is_better:
                if part_obj_ind.value < part_obj_other_ind.value:
                    return False
                elif part_obj_ind.value > part_obj_other_ind.value:
                    better_in_at_least_one = True
            elif part_obj_ind.value > part_obj_other_ind.value:
                return False
            elif part_obj_ind.value < part_obj_other_ind.value:
                better_in_at_least_one = True

    return better_in_at_least_one


def start_ranking(
    population: List[MOIndividual],
) -> Tuple[
    List[MOIndividual],
    Dict[MOIndividual, List[MOIndividual]],
    Dict[MOIndividual, int],
]:
    rank_list = []
    dominated_by_ind = {}
    dominates_ind = {}
    for indiv in population:
        dominated_by_ind[indiv] = []
        dominates_ind[indiv] = 0
        for other_indiv in population:
            if other_indiv == indiv:
                continue
            if check_ind_dominance(indiv, other_indiv):
                dominated_by_ind[indiv].append(other_indiv)
            elif check_ind_dominance(other_indiv, indiv):
                dominates_ind[indiv] += 1
        if dominates_ind[indiv] == 0:
            rank_list.append(indiv)
    return rank_list, dominated_by_ind, dominates_ind
