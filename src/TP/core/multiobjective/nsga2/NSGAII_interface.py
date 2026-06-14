from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from TP.core.individuals.encoding import Encoder
from TP.core.logging.observer import EAObserver
from TP.core.logging.progress import EALogger
from TP.core.multiobjective.fitness import MOFitnessCalculator
from TP.core.multiobjective.individuals.factory import MOIndividualFactory
from TP.core.multiobjective.individuals.population import MOPopulation
from TP.core.multiobjective.individuals.representation import MOIndividual
from TP.core.multiobjective.nsga2.evaluator import NSGAEvaluator
from TP.core.multiobjective.selection.parents.operators import MOParentSelector
from TP.core.multiobjective.selection.survivors.operators import (
    MOSurvivorSelector,
)
from TP.core.state import EAState
from TP.core.utils.initialization import IndividualInitializer
from TP.core.variation.mutation import MutOperator
from TP.core.variation.recombination import RecombOperator


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class NSGAIIOrchestratorTemplate(ABC):
    pop_size: int

    encoder: Encoder
    parent_selector: MOParentSelector
    recombinator: RecombOperator
    survivor_selector: MOSurvivorSelector

    mutation_operator: MutOperator

    ind_initializer: IndividualInitializer
    observers: List[EAObserver]
    loggers: List[EALogger]

    # defaults
    fitness_calculator: Optional[MOFitnessCalculator] = None
    individual_factory: Optional[MOIndividualFactory] = None
    max_generations: Optional[int] = 1000
    n_parents: int = 2
    # n_offsprings: int = 2
    p_m: float = 0.1
    p_c: float = 0.7

    def __post_init__(self):
        self._configure_fitness()
        self._configure_factory()

    @abstractmethod
    def _configure_fitness(self):
        """Subclass defines fitness semantics"""

    def _configure_factory(self):
        if self.individual_factory is None:
            if self.fitness_calculator is None:
                raise RuntimeError('Fitness must be configured first')

            self.individual_factory = MOIndividualFactory(
                fitness_calculator=self.fitness_calculator,
                encoder=self.encoder,
            )

    @abstractmethod
    def generate_individual(self) -> MOIndividual:
        """
        Logic to create initial population

        Returns
        -------
        Population
            Initial problem population.
        """
        pass

    def generate_initial_population(self) -> MOPopulation:
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
        population = MOPopulation(
            ind_list=pop_list,
        )
        return population

    def mutate(
        self, individuals_list: List[MOIndividual]
    ) -> List[MOIndividual]:
        """
        Abstract implementation of mutation

        Parameters
        ----------
        individuals_list : List[MOIndividual]
            list of individuals after recombination to be mutated.

        Returns
        -------
        List[MOIndividual]
            Individuals after recombination and mutation
        """
        if self.individual_factory is None:
            raise ValueError(
                'Individual factory missing to run mutation step.'
            )
        return [
            self.individual_factory.create(self.mutation_operator.execute(ind))
            for ind in individuals_list
        ]

    def select_next_generation(
        self,
        mutated_offsprings: List[MOIndividual],
        population: MOPopulation,
        state: EAState,
    ) -> MOPopulation:
        """
        Abstract implementation on how to create new population

        Parameters
        ----------
        mutated_offsprings : List[MOIndividual]
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
            state=state,
        )
        return next_gen

    @abstractmethod
    def stop_criteria(self, state: EAState) -> bool:
        """
        Abstraction of algorithm stop criteria. Defaults to 1000 generations

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
        max_gen = 1000
        if state.generation <= max_gen:
            return True
        return False

    def select_parents(
        self, population: MOPopulation, state: EAState
    ) -> List[MOIndividual]:
        """
        Method to select parents.

        Returns
        -------
        List[MOIndividual]
            list of selected parents individuals
        """
        parents_list = self.parent_selector.select_parents(
            num_parents=self.n_parents,
            pop=population,
            state=state,
        )
        return parents_list

    def recombine_parents(self, parents_list) -> List[MOIndividual]:
        """
        Method to execute crossover.

        Returns
        -------
        List[MOIndividual]
            List of individuals after crossover
        """
        offsprings_chrm_list = self.recombinator.recombine(
            parents_list=parents_list,
            p_c=self.p_c,
        )
        if self.individual_factory is not None:
            offsprings_ind_list = [
                self.individual_factory.create(chrm)
                for chrm in offsprings_chrm_list
            ]
        else:
            raise ValueError(
                'Individual factory attribut must be different from None ',
                'to execute recombination',
            )

        return offsprings_ind_list

    def generate_offsprings(
        self,
        total_offsprings: int,
        state: EAState,
    ) -> List[MOIndividual]:
        """
        Method to generate offsprings.

        Returns
        -------
        MOIndividual
            Most relevant MOIndividual from population
        """

        if total_offsprings is None:
            total_offsprings = state.population.size
        assert total_offsprings % self.recombinator.n_offsprings == 0

        offspings_ind_list = []

        while len(offspings_ind_list) < total_offsprings:
            parents_list = self.parent_selector.select_parents(
                num_parents=self.recombinator.n_parents,
                pop=state.population,
                state=state,
            )
            children_chrm_list = self.recombinator.recombine(
                parents_list=parents_list,
                p_c=self.p_c,
            )
            if self.individual_factory is None:
                raise ValueError(
                    'Individual factory attribut must be different from None ',
                    'to execute recombination',
                )

            for chrm in children_chrm_list:
                offspings_ind_list.append(self.individual_factory.create(chrm))

        return offspings_ind_list

    @abstractmethod
    def get_output(self, state: EAState) -> List[MOIndividual]:
        """
        Abstract method to get output from population after stop
        criteria is reached.

        Returns
        -------
        MOIndividual
            Most relevant MOIndividual from population
        """
        if state.fronts is not None:
            return state.fronts[0]

    @abstractmethod
    def run(self):
        population: MOPopulation = self.generate_initial_population()
        evaluator = NSGAEvaluator()

        state: EAState = evaluator.evaluate(population, generation=0)

        self._notify_start(state)

        while self.stop_criteria(state):
            offsprings = self.generate_offsprings(
                state.population, self.n_offsprings, state=state
            )
            mutated_offsprings = self.mutate(offsprings)

            selection_population = MOPopulation(
                ind_list=mutated_offsprings + state.population.ind_list
            )

            updated_state = evaluator.evaluate(
                population=selection_population,
                generation=state.generation,
            )

            population = self.select_next_generation(
                mutated_offsprings=mutated_offsprings,
                population=population,
                state=updated_state,
            )

            updated_state = evaluator.evaluate(
                population=population,
                generation=state.generation,
            )
            state.generation += 1

            if state.generation % 50 == 0:
                self._notify_generation_end(state)

        self._notify_end(state)

        best_front = self.get_output(state)
        return best_front

    def _notify_start(self, state: EAState):
        for log in self.loggers or []:
            log.on_start(state)

    def _notify_generation_end(self, state):
        for obs in self.observers or []:
            obs.on_generation_end(state)
        for log in self.loggers or []:
            log.on_generation_end(state)

    def _notify_end(self, state):
        for log in self.loggers or []:
            log.on_end(state)
