from absl import app

import time
import random
import numpy as np

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

from scripted_agents import agent_utils as utils


class MoveToBeaconAgent(base_agent.BaseAgent):

    def __init__(self):
        super(MoveToBeaconAgent, self).__init__()
        self.score = None

    def step(self, obs):
        super(MoveToBeaconAgent, self).step(obs)

        x = random.randint(0, 63)
        y = random.randint(0, 63)
        if utils.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
            return actions.FUNCTIONS.Move_minimap('now', (x, y))

        marine_units = utils.get_units_by_type(obs, units.Terran.Marine)
        marine_unit = random.choice(marine_units)

        return actions.FUNCTIONS.select_point('select_all_type', (marine_unit.x, marine_unit.y))

        # return actions.FUNCTIONS.no_op()


def main(unused):
    agent = MoveToBeaconAgent()

    num_episodes = 20

    rewards_sum_list = []
    total_episode_reward_list = []

    try:
        for i_episode in range(1, num_episodes + 1):

            rewards_sum = []
            total_reward = 0

            with sc2_env.SC2Env(
                    map_name="MoveToBeacon",
                    players=[sc2_env.Agent(sc2_env.Race.terran)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True),
                    step_mul=16,  # the default value is 8, this is also suggested value from the paper
                    game_steps_per_episode=0,  # makes the game for as long as necessary
                    visualize=False) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                time_steps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(time_steps[0])]

                    if time_steps[0].last():
                        total_episode_reward_list.append(total_reward)
                        print(f'total reward for episode {total_reward}')
                        break
                    time_steps = env.step(step_actions)
                    reward = time_steps[0].reward
                    total_reward = total_reward + reward

        print(f'in {num_episodes} episodes the agent got a total reward of {np.sum(total_episode_reward_list)}')

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
