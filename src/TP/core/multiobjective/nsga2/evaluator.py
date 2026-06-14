from TP.core.multiobjective.ranking.pareto_diversity import (
    fast_non_dominated_sort,
)
from TP.core.multiobjective.ranking.sort import compute_crowding_distance
from TP.core.state import EAState


class NSGAEvaluator:
    @staticmethod
    def evaluate(population, generation=0) -> EAState:

        fronts = fast_non_dominated_sort(population)

        rank = {ind: i for i, front in enumerate(fronts) for ind in front}

        crowding = {}
        for front in fronts:
            crowding.update(compute_crowding_distance(front))

        total_fronts = 0
        for front in fronts:
            total_fronts += len(front)

        return EAState(
            population=population,
            generation=generation,
            fronts=fronts,
            rank=rank,
            crowding=crowding,
            multi_objective=True,
        )
