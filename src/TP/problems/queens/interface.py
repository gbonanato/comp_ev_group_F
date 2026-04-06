from abc import abstractmethod
from typing import Optional

from pydantic.dataclasses import dataclass

from TP.core.fitness import FitnessCalculator
from TP.core.interface import OrchestratorTemplate
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
from TP.problems.queens.individuals.factory import PermIndividualFactory
from TP.problems.queens.individuals.population import QueensPopulation
from TP.problems.queens.individuals.representation import QueensIndividual
from TP.problems.queens.utils.initialization import RandomPermInitilizer
from TP.problems.queens.variation.mutation import QueenSwapMutation


@dataclass
class QueenProblemOrchestrator(OrchestratorTemplate):
    board_size: int
    pop_size: int
    n_offsprings: Optional[int]

    # defaults
    parent_selector: ParentSelector = RouletteStrategy()
    recombinator: RecombOperator = PMX()
    survivor_selector: SurvivorSelector = ElitismGenerational()

    mutation_operator: MutOperator = QueenSwapMutation()
    fitness_calculator: FitnessCalculator = QueensBoardFitness()

    ind_initializer: RandomPermInitilizer = RandomPermInitilizer()

    p_m: float = 0.2
    p_c: float = 0.5

    def generate_individual(self) -> QueensIndividual:
        chrm = self.ind_initializer.generate_chrm(self.board_size)
        return PermIndividualFactory(
            chrm,
            fitness_calculator=self.fitness_calculator,
            mutation_operator=self.mutation_operator,
            p_m=self.p_m,
        ).create()

    @abstractmethod
    def stop_criteria(state: EAState) -> bool:
        """
        ABstraction of algorithm stop criteria

        Parameters
        ----------
        state : EAState
            State class that keeps track of relevant information
            to reach stop criteria.

        Returns
        -------
        bool
            Decision to stop or not the evolutionary iterations
        """
        pass

    @abstractmethod
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


# TODO: definir stop criteria e definir get output.
