import sc2

from sc2 import run_game, maps, BotAI

from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.player import Bot, Computer

class SimpleAgent(BotAI):

    def __init__(self):
        super().__init__()
        self.actions = []

    async def on_step(self, iteration: int):
        self.actions = []

        if iteration == 0:
            tar