from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic.dataclasses import dataclass

from TP.core.fitness import FitnessCalculator
from TP.core.individuals.factory import IndividualFactory
from TP.core.individuals.population import Population
from TP.core.individuals.representation import Individual
from TP.core.selection.parents.operators import ParentSelector
from TP.core.selection.survivors.operators import SurvivorSelector
from TP.core.state import EAState
from TP.core.utils.initialization import IndividualInitializer
from TP.core.variation.mutation import MutOperator
from TP.core.variation.recombination import RecombOperator


@dataclass
class OrchestratorTemplate(ABC):
    pop_size: int
    n_offsprings: Optional[int]

    # defaults
    parent_selector: ParentSelector
    recombinator: RecombOperator
    survivor_selector: SurvivorSelector

    mutation_operator: MutOperator
    fitness_calculator: FitnessCalculator

    ind_initializer: IndividualInitializer

    p_m: float = 0.1
    p_c: float = 0.7

    @abstractmethod
    def generate_individual(self) -> Individual:
        """
        Logic to create initial population

        Returns
        -------
        Population
            Initial problem population.
        """
        pass

    def generate_initial_population(self) -> Population:
        """
        Logic to create initial population

        Returns
        -------
        Population
            Initial problem population.
        """
        pop_list = []
        for _ in range(self.pop_size):
            ind = self.generate_individual()
            pop_list.append(ind)
        population = Population(
            ind_list=pop_list,
            parent_selector=self.parent_selector,
            recombinator=self.recombinator,
            survivor_selector=self.survivor_selector,
            indiv_factory=IndividualFactory(
                fitness_calculator=self.fitness_calculator,
                mutation_operator=self.mutation_operator,
                p_m=self.p_m,
            ),
        )
        return population

    def mutate(individuals_list: List[Individual]) -> List[Individual]:
        """
        Abstract implementation of mutation

        Parameters
        ----------
        individuals_list : List[Individual]
            list of individuals after recombination to be mutated.

        Returns
        -------
        List[Individual]
            Individuals after recombination and mutation
        """
        return [ind.mutate() for ind in individuals_list]

    def select_next_generation(
        mutated_offsprings: List[Individual],
        population: Population,
    ) -> Population:
        """
        Abstract implementation on how to create new population

        Parameters
        ----------
        mutated_offsprings : List[Individual]
            offsprings after reccombination and mutation
        population : Population
            Original population

        Returns
        -------
        Population
            Next generation population
        """
        next_gen = population.select_next_generation(
            mutated_offsprings,
            population.size,
        )
        return next_gen

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

    def generate_offsprings(self, population) -> Individual:
        """
        Method to generate offsprings.

        Returns
        -------
        Individual
            Most relevant individual from population
        """
        offsprings = population.generate_offsprings(self.n_offsprings)
        return offsprings

    @abstractmethod
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

    @abstractmethod
    def run(self):
        population = self.generate_initial_population()

        state = EAState(
            population=population,
            generation=0,
        )

        while self.stop_criteria(state):
            offsprings = self.generate_offsprings(
                population,
                self.n_offsprings,
            )
            mutated_offsprings = self.mutate(offsprings)

            population = self.select_next_generation(
                mutated_offsprings,
                population.size,
            )
            state.population = population
            state.generation += 1

        output_individual = self.get_output(population)
        return output_individual.decode()
