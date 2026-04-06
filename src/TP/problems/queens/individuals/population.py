from typing import List, Optional

from TP.core.individuals.population import Population
from TP.problems.queens.individuals.representation import QueensIndividual


class QueensPopulation(Population):
    def generate_offsprings(
        self,
        n_offsprings: int = None,
    ) -> List[QueensIndividual]:

        if n_offsprings is None:
            n_offsprings = len(self.ind_list)
        assert n_offsprings % self.recombinator.n_offsprings == 0

        offspings_list = []

        while len(offspings_list) < n_offsprings:
            parents_list = self.parent_selector.select_parents(
                self.ind_list, self.recombinator.n_parents
            )
            children = self.recombinator.recombine(parents_list)

            for child in children:
                offspings_list.append(self.indiv_factory(child))

        return offspings_list

    def select_next_generation(
        self,
        offspings_list: List[QueensIndividual],
        n_survivors: Optional[int],
    ):
        if n_survivors is None:
            n_survivors = self.size
        next_gen = self.survivor_selector.select_survivors(
            self.ind_list,
            offspings_list,
            n_survivors,
        )
        return next_gen
