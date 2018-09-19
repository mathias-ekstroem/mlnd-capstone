from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

import random

import scripted_agents.agent_utils as util


class TerranAgent(base_agent.BaseAgent):

    def __init__(self):
        super(TerranAgent, self).__init__()
        self.n_supply_depots = 0
        self.n_scvs = 0
        self.idle_worker_selected = False

    def step(self, obs):
        super(TerranAgent, self).step(obs)

        # get the idle worker back to mining
        if self.idle_worker_selected:
            self.idle_worker_selected = False
            if util.can_do(obs, actions.FUNCTIONS.select_point.id):
                # get location of minerals
                obs.observation.feature_screen

        # select idle worker and return them to mining
        if util.can_do(obs, actions.FUNCTIONS.select_idle_worker.id):
            self.idle_worker_selected = True
            return actions.FUNCTIONS.select_idle_worker()

        # build a supply depot
        if util.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
            # select the point at which to build the supply depot
            x = random.randint(1, 83)
            y = random.randint(1, 83)

            return actions.FUNCTIONS.Build_SupplyDepot_screen('now', (x, y))

        # select scv's
        scv = util.get_random_unit_by_type(obs, units.Terran.SCV)
        if util.can_do(obs, actions.FUNCTIONS.select_point.id):
            return actions.FUNCTIONS.select_point('select', (scv.x, scv.y))

        return actions.FUNCTIONS.no_op()


def main(unused):
    agent = TerranAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="AbyssalReef",
                    players=[sc2_env.Agent(sc2_env.Race.terran),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True),
                    step_mul=16,  # the default value is 8, this is also suggested value from the paper
                    game_steps_per_episode=0,  # makes the game for as long as necessary
                    visualize=True) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                time_steps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(time_steps[0])]
                    if time_steps[0].last():
                        break
                    time_steps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
