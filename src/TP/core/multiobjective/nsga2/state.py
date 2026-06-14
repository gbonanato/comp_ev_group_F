# from typing import Dict, List

# from pydantic.dataclasses import dataclass

# from TP.core.multiobjective.individuals.population import MOPopulation
# from TP.core.multiobjective.individuals.representation import MOIndividual


# @dataclass
# class EAState:
#     population: MOPopulation
#     fronts: List[List[MOIndividual]]
#     rank: Dict[MOIndividual, int]  # Best rank is zero.
#     crowding: Dict[MOIndividual, float]
#     generation: int = 0
#     feasibility: bool = False

#     # n_evaluations: int
#     # best_fitness: float
#     # start_time: float
#     # diversity: float
#     # no_improv_counter: int
