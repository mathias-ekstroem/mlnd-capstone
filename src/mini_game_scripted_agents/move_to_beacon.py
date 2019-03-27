import random
from collections import deque

import numpy as np
import tensorflow as tf
from pysc2.agents import base_agent
from pysc2.lib import actions, units, features

from scripted_agents import agent_utils as utils


# maybe this one can be used for sampling random actions from the environment
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


class MoveToBeaconSmartAgent(base_agent.BaseAgent):
    _NO_OP = actions.FUNCTIONS.no_op()
    _PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL
    FUNCTIONS = actions.FUNCTIONS

    def __init__(self):
        super(MoveToBeaconSmartAgent, self).__init__()
        self.score = None
        self.FUNCTIONS = actions.FUNCTIONS
        self.rewards = None

    def step(self, obs):
        super(MoveToBeaconSmartAgent, self).step(obs)
        if self.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            pass
        else:
            return self.FUNCTIONS.select_army('select')


class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4,
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
