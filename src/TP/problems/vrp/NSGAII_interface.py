import random
from typing import List, Optional

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from TP.core.heuristics.sequence import run_2opt
from TP.core.individuals.encoding import PermutationEncoder
from TP.core.logging.observer import EAObserver
from TP.core.logging.progress import EALogger
from TP.core.multiobjective.fitness import MOFitnessCalculator
from TP.core.multiobjective.individuals.factory import MOIndividualFactory
from TP.core.multiobjective.individuals.population import MOPopulation
from TP.core.multiobjective.individuals.representation import MOIndividual
from TP.core.multiobjective.nsga2.evaluator import NSGAEvaluator
from TP.core.multiobjective.nsga2.NSGAII_interface import (
    NSGAIIOrchestratorTemplate,
)
from TP.core.multiobjective.selection.parents.operators import (
    CrowdTournament,
    MOParentSelector,
)
from TP.core.multiobjective.selection.survivors.operators import (
    MOSurvivorSelector,
    NonDominanceOrdering,
)
from TP.core.state import EAState
from TP.core.utils.initialization import IndividualInitializer
from TP.core.variation.mutation import RSM, MutOperator
from TP.core.variation.recombination import RecombOperator
from TP.problems.tsp.utils.initialization import TSPInitilizer
from TP.problems.tsp.variation.recombination import SCX
from TP.problems.vrp.individuals.encoding import VRPEncoder
from TP.problems.vrp.multiobjective.fitness import VRPFitness
from TP.problems.vrp.utils.problem_data import VRPProblemData


@dataclass(config=ConfigDict(arbitrary_types_allowed=True), kw_only=True)
class VRPNSGAIIOrchestrator(NSGAIIOrchestratorTemplate):
    problem_instance: VRPProblemData
    pop_size: int

    encoder: PermutationEncoder = Field(default_factory=VRPEncoder)
    parent_selector: MOParentSelector = Field(default_factory=CrowdTournament)
    recombinator: RecombOperator = Field(default_factory=SCX)
    survivor_selector: MOSurvivorSelector = Field(
        default_factory=NonDominanceOrdering
    )

    mutation_operator: MutOperator = Field(default_factory=RSM)

    ind_initializer: IndividualInitializer = Field(
        default_factory=TSPInitilizer
    )
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
    p_ls: float = 0.3

    def _configure_fitness(self):
        self.fitness_calculator = VRPFitness(
            problem_instance=self.problem_instance
        )

    def __post_init__(self):
        super().__post_init__()

    def generate_individual(self) -> MOIndividual:
        problem_size = len(self.problem_instance.dist_mtx)
        chrm = self.ind_initializer.generate_chrm(problem_size)
        if self.individual_factory is not None:
            individual = self.individual_factory.create(chrm)
            return individual
        else:
            raise ValueError(
                'Individual Factory muts be instatiated to generate individual'
            )

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
        max_gen = self.max_generations
        if state.generation <= max_gen:
            return True
        return False

    def heuristic_2_opt(
        self,
        state: EAState,
        problem_instace: VRPProblemData,
        p_ls: float = 0.3,
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
        if state.fronts is None:
            raise ValueError('State missing fronts to run heuristics')
        for ind in state.fronts[0]:
            if random.random() < p_ls:
                ind.chrm = run_2opt(ind, dist_mtx=problem_instace.dist_mtx)
                ind._fitness = None

    def get_output(self, state: EAState) -> List[MOIndividual]:
        """
        Abstract method to get output from population after stop
        criteria is reached.

        Returns
        -------
        MOIndividual
            Most relevant MOIndividual from population
        """
        if state.fronts is None:
            raise ValueError(
                'At least one individual must populate the first front.'
            )

        return state

    def run(self):
        population: MOPopulation = self.generate_initial_population()
        evaluator = NSGAEvaluator()

        state: EAState = evaluator.evaluate(population, generation=0)

        self._notify_start(state)

        while self.stop_criteria(state):
            offsprings = self.generate_offsprings(
                total_offsprings=population.size, state=state
            )
            mutated_offsprings = self.mutate(offsprings)

            selection_population = MOPopulation(
                ind_list=mutated_offsprings + population.ind_list
            )

            updated_state = evaluator.evaluate(
                population=selection_population,
                generation=state.generation,
            )

            updated_pop = self.select_next_generation(
                mutated_offsprings=mutated_offsprings,
                population=population,
                state=updated_state,
            )

            state = evaluator.evaluate(
                population=updated_pop,
                generation=state.generation,
            )

            self.heuristic_2_opt(
                state=state,
                problem_instace=self.problem_instance,
                p_ls=self.p_ls,
            )

            state = evaluator.evaluate(
                population=population,
                generation=state.generation,
            )

            state.generation += 1

            if state.generation % 100 == 0:
                self._notify_generation_end(state)

        self._notify_end(state)

        best_front = self.get_output(state)
        return best_front
