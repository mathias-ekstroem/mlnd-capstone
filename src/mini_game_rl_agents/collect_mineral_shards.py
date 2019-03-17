import random

from pysc2.agents import base_agent
from pysc2.lib import actions, units

from scripted_agents import agent_utils as utils

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point




class MoveToBeaconSmartAgent(base_agent.BaseAgent):

    def __init__(self):
        super(MoveToBeaconRandomAgent, self).__init__()
        self.score = None

    def step(self, obs):
        super(MoveToBeaconRandomAgent, self).step(obs)

        x = random.randint(0, 63)
        y = random.randint(0, 63)
        if utils.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
            return actions.FUNCTIONS.Move_minimap('now', (x, y))

        marine_units = utils.get_units_by_type(obs, units.Terran.Marine)
        marine_unit = random.choice(marine_units)

        return actions.FUNCTIONS.select_point('select_all_type', (marine_unit.x, marine_unit.y))
