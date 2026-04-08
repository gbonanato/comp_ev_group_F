from typing import List, Optional

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from TP.core.fitness import FitnessCalculator
from TP.core.interface import OrchestratorTemplate
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
from TP.core.variation.mutation import MutOperator
from TP.core.variation.recombination import PMX, RecombOperator
from TP.problems.queens.fitness import QueensBoardFitness
from TP.problems.queens.individuals.population import QueensPopulation
from TP.problems.queens.individuals.representation import QueensIndividual
from TP.problems.queens.utils.initialization import RandomPermInitilizer
from TP.problems.queens.variation.mutation import QueenSwapMutation


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class QueenProblemOrchestrator(OrchestratorTemplate):
    board_size: int
    pop_size: int
    n_offsprings: Optional[int]  # Per recombination

    # defaults
    parent_selector: ParentSelector = Field(default_factory=RouletteStrategy)
    recombinator: RecombOperator = Field(default_factory=PMX)
    survivor_selector: SurvivorSelector = Field(
        default_factory=ElitismGenerational
    )

    mutation_operator: MutOperator = Field(default_factory=QueenSwapMutation)
    fitness_calculator: FitnessCalculator = Field(
        default_factory=QueensBoardFitness
    )

    ind_initializer: RandomPermInitilizer = Field(
        default_factory=RandomPermInitilizer
    )

    observers: List[EAObserver] = Field(default_factory=list)
    loggers: List[EALogger] = Field(default_factory=list)

    p_m: float = 0.2
    p_c: float = 0.5

    def generate_individual(self) -> QueensIndividual:
        chrm = self.ind_initializer.generate_chrm(self.board_size)
        return QueensIndividual(chrm, self.fitness_calculator)

    @staticmethod
    def stop_criteria(state: EAState) -> bool:
        if state.feasibility:
            return True
        return False

    @staticmethod
    def check_feasibility(
        population: QueensPopulation,
    ) -> bool:
        for ind in population:
            if ind.fitness == 0:
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
            feasible = self.check_feasibility(population)
            state.population = population
            state.generation += 1
            state.feasibility = feasible

            self._notify_generation_end(state)

        self._notify_end(state)
        output_individual = self.get_output(population)
        return output_individual

    @staticmethod
    def get_output(population: QueensPopulation) -> QueensIndividual:
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
