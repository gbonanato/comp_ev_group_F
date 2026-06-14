from typing import Dict, List, Optional, Union

from pydantic.dataclasses import dataclass

from TP.core.individuals.population import Population
from TP.core.multiobjective.individuals.population import MOPopulation
from TP.core.multiobjective.individuals.representation import MOIndividual


@dataclass
class EAState:
    population: Union[Population, MOPopulation]
    generation: int = 0
    feasibility: bool = False
    fronts: Optional[List[List[MOIndividual]]] = None
    rank: Optional[Dict[MOIndividual, int]] = None  # Best rank is zero.
    crowding: Optional[Dict[MOIndividual, float]] = None
    multi_objective: bool = False
    # n_evaluations: int
    # best_fitness: float
    # start_time: float
    # diversity: float
    # no_improv_counter: int


# TODO: IMPLEMENT TRACKERS FOR OTHER STOP CRITERION
