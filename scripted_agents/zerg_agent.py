import random

from absl import app
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

import scripted_agents.agent_utils as util


class ZergAgent(base_agent.BaseAgent):

    def __init__(self):
        super(ZergAgent, self).__init__()

        self.attack_coordinates = None

    def step(self, obs):
        super(ZergAgent, self).step(obs)

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()

            x_mean = player_x.mean()
            y_mean = player_y.mean()

            if x_mean <= 31 and y_mean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)

        # select army of zerglings in order to attack
        zerglings = util.get_units_by_type(obs, units.Zerg.Zergling)
        if len(zerglings) > 10:
            # if we selected our army of zerglings in the previous step we can now attack at the defined coordinates
            if util.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                return actions.FUNCTIONS.Attack_minimap('now', self.attack_coordinates)

            if util.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army('select')

        # build exactly one spawning pool
        spawning_pools = util.get_units_by_type(obs, units.Zerg.SpawningPool)
        if len(spawning_pools) == 0:
            if util.unit_type_is_selected(obs, units.Zerg.Drone):
                # check the build spawning pool action is available, otherwise the environment will throw an exception
                if util.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                    # select a random point on screen to build the spawning pool
                    # hopefully if will be in a place with some creep
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)

                    return actions.FUNCTIONS.Build_SpawningPool_screen('now', (x, y))

            # get a list of drones via the feature_units
            drones = util.get_units_by_type(obs, units.Zerg.Drone)

            if len(drones) > 0:
                drone = random.choice(drones)

                """
                there's more properties accesible than just the x and y coordinate these include
                health, shields, energy, build_progress, and ideal_harvesters + assigned_harvesters
                now select the drone, actually selects all drones because of the "select_all_type"
                """
                return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))

        # build overlord if we've run of unit capacity
        free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
        if free_supply == 0:
            if util.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                return actions.FUNCTIONS.Train_Overlord_quick('now')

        # build zerglings
        if util.unit_type_is_selected(obs, units.Zerg.Larva):
            # check we have enough resources and unit capacity to build a zergling
            if util.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                return actions.FUNCTIONS.Train_Zergling_quick('now')

        # select available larvae
        larvae = util.get_units_by_type(obs, units.Zerg.Larva)
        if len(larvae) > 0:
            larva = random.choice(larvae)
            return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))

        # should make this agent work, but not do anything
        return actions.FUNCTIONS.no_op()


def main(unused):
    agent = ZergAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="AbyssalReef",
                    players=[sc2_env.Agent(sc2_env.Race.zerg),
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
