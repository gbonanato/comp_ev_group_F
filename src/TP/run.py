from TP.problems.queens.interface import QueenProblemOrchestrator

orchestrator = QueenProblemOrchestrator(
    board_size=8,
    pop_size=20,
)

test = orchestrator.run()

print(test)
