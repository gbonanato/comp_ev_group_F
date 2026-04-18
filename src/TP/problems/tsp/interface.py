from typing import List, Optional

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from tsplib95.models import StandardProblem

from TP.core.GA_interface import GAOrchestratorTemplate
from TP.core.individuals.encoding import PermutationEncoder
from TP.core.individuals.population import Population
from TP.core.individuals.representation import Individual
from TP.core.logging.observer import EAObserver
from TP.core.logging.progress import EALogger
from TP.core.selection.parents.operators import (
    ParentSelector,
    RouletteStrategy,
)
from TP.core.selection.survivors.operators import (
    ElitismGenerational,
    SurvivorSelector,
)
from TP.core.state import EAState
from TP.core.utils.initialization import IndividualInitializer
from TP.core.variation.mutation import RSM, MutOperator
from TP.core.variation.recombination import PMX, RecombOperator
from TP.problems.tsp.fitness import TSPFitness
from TP.problems.tsp.individuals.encoding import TSPEncoder
from TP.problems.tsp.utils.initialization import TSPInitilizer


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class TSPOrchestrator(GAOrchestratorTemplate):
    problem_instance: StandardProblem
    pop_size: int
    n_offsprings: Optional[int]  # Per recombination

    # defaults
    encoder: PermutationEncoder = Field(default_factory=TSPEncoder)
    parent_selector: ParentSelector = Field(default_factory=RouletteStrategy)
    recombinator: RecombOperator = Field(default_factory=PMX)
    survivor_selector: SurvivorSelector = Field(
        default_factory=ElitismGenerational
    )

    mutation_operator: MutOperator = Field(default_factory=RSM)

    ind_initializer: IndividualInitializer = Field(
        default_factory=TSPInitilizer
    )

    observers: List[EAObserver] = Field(default_factory=list)
    loggers: List[EALogger] = Field(default_factory=list)

    p_m: float = 0.1
    p_c: float = 0.7

    def _configure_fitness(self):
        self.fitness_calculator = TSPFitness(
            problem_instance=self.problem_instance
        )

    def __post_init__(self):
        super().__post_init__()

    def generate_individual(self) -> Individual:
        problem_size = self.problem_instance.dimension
        chrm = self.ind_initializer.generate_chrm(problem_size)
        individual = self.individual_factory.create(chrm)
        return individual

    def stop_criteria(self, state: EAState) -> bool:
        if state.generation == self.max_generations:
            return True
        return False

    def run(self):
        population = self.generate_initial_population()

        state = EAState(
            population=population,
            generation=0,
            feasibility=False,
        )

        self._notify_start(state)

        while not self.stop_criteria(state):
            offsprings = self.generate_offsprings(
                population,
            )
            mutated_offsprings = self.mutate(offsprings)

            population = self.select_next_generation(
                mutated_offsprings=mutated_offsprings,
                population=population,
            )
            state.population = population
            state.generation += 1

            self._notify_generation_end(state)

        self._notify_end(state)
        output_individual = self.get_output(population)

        return output_individual

    @staticmethod
    def get_output(population: Population) -> Individual:
        """
        Abstract method to get output from population after stop
        criteria is reached.

        Returns
        -------
        Individual
            Most relevant individual from population
        """
        most_fit = [indiv.fitness for indiv in population.ind_list]
        most_fit_position = most_fit.index(max(most_fit))
        return population.ind_list[most_fit_position]
