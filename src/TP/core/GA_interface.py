from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from TP.core.fitness import FitnessCalculator
from TP.core.individuals.population import Population
from TP.core.individuals.representation import Individual
from TP.core.logging.observer import EAObserver
from TP.core.logging.progress import EALogger
from TP.core.selection.parents.operators import ParentSelector
from TP.core.selection.survivors.operators import SurvivorSelector
from TP.core.state import EAState
from TP.core.utils.initialization import IndividualInitializer
from TP.core.variation.mutation import MutOperator
from TP.core.variation.recombination import RecombOperator


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class GAOrchestratorTemplate(ABC):
    pop_size: int

    parent_selector: ParentSelector
    recombinator: RecombOperator
    survivor_selector: SurvivorSelector

    mutation_operator: MutOperator
    fitness_calculator: FitnessCalculator

    ind_initializer: IndividualInitializer
    observers: List[EAObserver]
    loggers: List[EALogger]

    # defaults
    n_parents: Optional[int] = 2
    n_offsprings: Optional[int] = 2
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
        )
        return population

    def mutate(self, individuals_list: List[Individual]) -> List[Individual]:
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
        return [
            self.mutation_operator.execute(ind) for ind in individuals_list
        ]

    def select_next_generation(
        self,
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
        next_gen = self.survivor_selector.select_survivors(
            parents=population.ind_list,
            offsprings=mutated_offsprings,
            n_survivors=population.size,
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

    def select_parents(self, population: Population) -> List[Individual]:
        """
        Method to select parents.

        Returns
        -------
        List[Individual]
            list of selected parents individuals
        """
        parents_list = self.parent_selector.select_parents(
            num_parents=self.n_parents,
            pop=population,
        )
        return parents_list

    def recombine_parents(self, parents_list) -> List[Individual]:
        """
        Method to execute crossover.

        Returns
        -------
        List[Individual]
            List of individuals after crossover
        """
        offsprings_list = self.recombinator.recombine(
            parents_list=parents_list,
            fitness_calculator=self.fitness_calculator,
            p_c=self.p_c,
        )
        return offsprings_list

    def generate_offsprings(
        self,
        population: Population,
        total_offsprings: int = None,
    ) -> Individual:
        """
        Method to generate offsprings.

        Returns
        -------
        Individual
            Most relevant individual from population
        """

        if total_offsprings is None:
            total_offsprings = population.size
        assert total_offsprings % self.recombinator.n_offsprings == 0

        offspings_list = []

        while len(offspings_list) < total_offsprings:
            parents_list = self.parent_selector.select_parents(
                num_parents=self.recombinator.n_parents,
                pop=population,
            )
            children = self.recombinator.recombine(
                parents_list=parents_list,
                fitness_calculator=self.fitness_calculator,
                p_c=self.p_c,
            )

            for child in children:
                offspings_list.append(child)

        return offspings_list

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

        self._notify_start(state)

        while self.stop_criteria(state):
            offsprings = self.generate_offsprings(
                population,
                self.n_offsprings,
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

    def _notify_start(self, state):
        for obs in self.observers or []:
            obs.on_start(state)
        for log in self.loggers or []:
            log.on_start(state)

    def _notify_generation_end(self, state):
        for obs in self.observers or []:
            obs.on_generation_end(state)
        for log in self.loggers or []:
            log.on_generation_end(state)

    def _notify_end(self, state):
        for obs in self.observers or []:
            obs.on_end(state)
        for log in self.loggers or []:
            log.on_end(state)
