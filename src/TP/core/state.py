from pydantic.dataclasses import dataclass

from TP.core.individuals.population import Population


@dataclass
class EAState:
    population: Population
    generation: int
    feasibility: bool
    # n_evaluations: int
    # best_fitness: float
    # start_time: float
    # diversity: float
    # no_improv_counter: int


# TODO: IMPLEMENT TRACKERS FOR OTHER STOP CRITERION
