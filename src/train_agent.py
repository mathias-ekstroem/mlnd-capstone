import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features

from mini_game_scripted_agents.move_to_beacon import MoveToBeaconRandomAgent, MoveToBeaconSmartAgent, QNetwork, Memory

# MoveToBeacon, DefeatZerglingsAndBanelings, CollectMineralShards
# BuildMarines, CollectMineralsAndGas, DefeatRoaches, FindAndDefeatZerglings

max_mem_size = 1000
learning_rate = 0.01
step_mul = 16  # the default value is 8, this is also suggested value from the paper

pre_train_length = 20  # number of episodes for pre training

screen_dims = (84, 84)
minimap_dims = (64, 64)
num_episodes = 50

explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

gamma = 0.99                   # future reward discount

def main(unused):
    agent = MoveToBeaconSmartAgent()

    rewards_in_episode_list = []
    total_episode_reward_list = []

    state_size = np.zeros(screen_dims).reshape(-1).shape[0]
    action_size = np.zeros(screen_dims).reshape(-1).shape[0]

    tf.reset_default_graph()  # dunno if I need to do this
    q_network = QNetwork(learning_rate, state_size, action_size)

    try:
        with sc2_env.SC2Env(
                map_name="MoveToBeacon",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True),
                step_mul=step_mul,
                game_steps_per_episode=0,  # makes the game for as long as necessary
                visualize=False) as env:
            print('starting pre memory population')
            memory = pre_populate_memory(env, pre_train_length, max_mem_size)
            print('memory pre populated')

            print('starting training')
            agent.setup(env.observation_spec(), env.action_spec())

            for i_episode in range(1, num_episodes + 1):
                time_steps = env.reset()
                step = time_steps[0]
                agent.reset()
                rewards_in_episode = []
                total_reward = 0

                while True:
                    step_actions = [agent.step(step)]

                    if step.last():
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


# pre populate the experience memory
def pre_populate_memory(env_, pre_train_length_, max_mem_size_):
    memory_ = Memory(max_mem_size_)
    # use the random action agent for pre populate
    random_agent = MoveToBeaconRandomAgent()

    for i_episode in range(1, pre_train_length_ + 1):
        time_steps = env_.reset()
        state = time_steps[0]  # list because of step multiplier
        random_agent.reset()
        total_reward = 0

        while True:
            step_actions = [random_agent.step(state)]
            time_steps = env_.step(step_actions)
            next_state = time_steps[0]
            reward = next_state.reward

            # create stuff that we need for experience memory
            screen = state.observation.feature_screen  # contains all feature layer (17)
            current_player_relative = screen.player_relative  # the feature layer that this agent should be using
            next_player_relative = next_state.observation.feature_screen.player_relative

            # get action vector for memory
            actions_ = step_actions[0]
            arguments_ = actions_.arguments[1]
            action_state_transf_ = np.zeros(screen_dims)
            action_state_transf_[arguments_[0]][arguments_[0]] = 1
            action_vector = action_state_transf_.reshape(-1).shape[0]

            if state.last():
                print(f'total reward for episode {i_episode}: {total_reward}')
                break

            total_reward = total_reward + reward
            memory_.add((current_player_relative, action_vector, reward, next_player_relative))
            state = next_state

    return memory_


if __name__ == "__main__":
    app.run(main)
