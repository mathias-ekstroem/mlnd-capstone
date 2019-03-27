import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, units, features

from scripted_agents import agent_utils as utils

# MoveToBeacon, DefeatZerglingsAndBanelings, CollectMineralShards
# BuildMarines, CollectMineralsAndGas, DefeatRoaches, FindAndDefeatZerglings

max_mem_size = 1000
learning_rate = 0.01
step_mul = 16  # the default value is 8, this is also suggested value from the paper

pre_train_length = 20  # number of episodes for pre training

screen_dims = (84, 84)
minimap_dims = (64, 64)
num_episodes = 50

explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0001  # exponential decay rate for exploration prob

gamma = 0.99  # future reward discount


def main(unused):
    rewards_in_episode_list = []
    total_episode_reward_list = []

    # state_size = np.zeros(screen_dims).reshape(-1).shape[0]
    # action_size = np.zeros(screen_dims).reshape(-1).shape[0]
    #
    tf.reset_default_graph()  # dunno if I need to do this
    # q_network = QNetwork(learning_rate, state_size, action_size)

    with tf.Session() as sess:
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

                print('creating agent')
                agent = DeepQNetworkAgent(sess, memory)
                sess.run(tf.global_variables_initializer())
                print('agent created')

                print('starting training')
                agent.setup(env.observation_spec(), env.action_spec())

                for i_episode in range(1, num_episodes + 1):
                    time_steps = env.reset()
                    state = time_steps[0]
                    agent.reset()
                    rewards_in_episode = []
                    total_reward = 0

                    while True:
                        step_actions = [agent.step(state)]
                        time_steps = env.step(step_actions)
                        next_state = time_steps[0]
                        reward = next_state.reward

                        if state.last():
                            total_episode_reward_list.append(total_reward)
                            rewards_in_episode_list.append(rewards_in_episode)
                            print(f'total reward for episode {i_episode}: {total_reward}')
                            break

                        total_reward = total_reward + reward
                        rewards_in_episode.append(rewards_in_episode)
                        agent.train()

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
            screen = state.observation.feature_screen  # contains all feature layers (17)
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


class DeepQNetworkAgent(base_agent.BaseAgent):

    def __init__(self, tf_session, memory, batch_size=20):
        super(DeepQNetworkAgent, self).__init__()
        self.tf_session = tf_session
        self.memory = memory
        self.batch_size = batch_size
        self.score = None

        # tf.reset_default_graph()  # dunno if I need to do this

        self.state_size = np.zeros(screen_dims).reshape(-1).shape[0]
        self.action_size = np.zeros(screen_dims).reshape(-1).shape[0]
        self.q_network = QNetwork(state_size=self.state_size, action_size=self.action_size)

    def step(self, obs):
        super(DeepQNetworkAgent, self).step(obs)

        # create stuff that we need for experience memory
        screen = obs.observation.feature_screen  # contains all feature layers (17)
        current_player_relative = screen.player_relative  # the feature layer that this agent should be using

        action_state_transf_ = np.zeros(screen_dims)
        action_state_transf_[arguments_[0]][arguments_[0]] = 1
        action_vector = action_state_transf_.reshape(-1).shape[0]


        if utils.can_do(obs, actions.FUNCTIONS.Move_screen.id):
            # do some logic here
            # reshape the state to same as in the memories
            feed = {self.q_network.inputs_: state.reshape((1, *state.shape))}
            Qs = self.tf_session.run(self.q_network.output, feed_dict=feed)
            action = np.argmax(Qs)

        marine_units = utils.get_units_by_type(obs, units.Terran.Marine)
        marine_unit = random.choice(marine_units)

        return actions.FUNCTIONS.select_point('select_all_type', (marine_unit.x, marine_unit.y))

    def train(self):
        batch = self.memory.sample(self.batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])

        # Train network
        target_Qs = self.tf_session.run(self.q_network.output, feed_dict={self.q_network.inputs_: next_states})

        # Set target_Qs to 0 for states where episode ends
        episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
        target_Qs[episode_ends] = (0, 0)

        targets = rewards + gamma * np.max(target_Qs, axis=1)

        loss, _ = self.tf_session.run([self.q_network.loss, self.q_network.opt],
                                      feed_dict={self.q_network.inputs_: states,
                                                 self.q_network.targetQs_: targets,
                                                 self.q_network.actions_: actions})


class MoveToBeaconRandomAgent(base_agent.BaseAgent):

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


class QNetwork:
    def __init__(self, learning_rate_=0.01, state_size=4,
                 action_size=2, hidden_size=10,
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

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
            self.opt = tf.train.AdamOptimizer(learning_rate_).minimize(self.loss)


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


if __name__ == "__main__":
    app.run(main)
