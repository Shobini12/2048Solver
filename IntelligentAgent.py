import random
from BaseAI import BaseAI
class IntelligentAgent(BaseAI):
def getMove(self, grid):
    # Selects a random move and returns it
    moveset = grid.getAvailableMoves()
    return random.choice(moveset)[0] if moveset else Non