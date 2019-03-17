from collections import deque

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from absl import app
from pysc2.env import sc2_env
from pysc2.lib import features


class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=(84, 84),
                 action_size=(84, 84), hidden_size=10,
                 name='QNetwork'):
        with tf.variable_scope(name):
            state_x, state_y = state_size
            action_x, action_y = action_size

            self.inputs_ = tf.placeholder(tf.float32, [None, state_x, state_y], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_x)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc2, action_size,
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)


class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


train_episodes = 1000  # max number of episodes to learn from
max_steps = 200  # max steps in an episode
gamma = 0.99  # future reward discount

# Exploration parameters
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # exponential decay rate for exploration prob

# Network parameters
hidden_size = 64  # number of units in each Q-network hidden layer
learning_rate = 0.0001  # Q-network learning rate

# Memory parameters
memory_size = 10000  # memory capacity
batch_size = 20  # experience mini-batch size
pre_train_length = batch_size  # number experiences to pretrain the memory

tf.reset_default_graph()
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate)


def pre_training(map_name):
    pass


def main(unused):
    # agent = MoveToBeaconRandomAgent()
    # agent = DefeatZerglingsAndBanelingsRandomAgent()
    # agent = CollectMineralShardsRandomAgent()

    num_episodes = 50

    rewards_in_episode_list = []
    total_episode_reward_list = []

    try:
        with sc2_env.SC2Env(
                map_name="FindAndDefeatZerglings",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True),
                step_mul=16,  # the default value is 8, this is also suggested value from the paper
                game_steps_per_episode=0,  # makes the game for as long as necessary
                visualize=False) as env:

            q_network = QNetwork(learning_rate=0.01, state_size=(84, 84), action_size=(84, 84))

            # agent.setup(env.observation_spec(), env.action_spec())

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
