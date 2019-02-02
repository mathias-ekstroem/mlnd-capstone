import matplotlib.pyplot as plt
import numpy as np
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features

from mini_game_scripted_agents.move_to_beacon import MoveToBeaconRandomAgent


def main(unused):
    agent = MoveToBeaconRandomAgent()

    num_episodes = 50

    rewards_in_episode_list = []
    total_episode_reward_list = []

    try:
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

            for i_episode in range(1, num_episodes + 1):

                time_steps = env.reset()
                agent.reset()

                rewards_in_episode = []
                total_reward = 0

                while True:
                    step_actions = [agent.step(time_steps[0])]

                    if time_steps[0].last():
                        total_episode_reward_list.append(total_reward)
                        rewards_in_episode_list.append(rewards_in_episode)
                        print(f'total reward for episode {i_episode}: {total_reward}')
                        break
                    time_steps = env.step(step_actions)
                    reward = time_steps[0].reward
                    total_reward = total_reward + reward
                    rewards_in_episode.append(rewards_in_episode)

        print(f'in {num_episodes} episodes the agent got a total reward of {np.sum(total_episode_reward_list)}')

        plt.plot(total_episode_reward_list)
        plt.savefig('move_to_beacon_random_rewards.png')

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
