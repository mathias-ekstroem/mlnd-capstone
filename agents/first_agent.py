import keras
import tensorflow as tf
import numpy as np
import random

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

from ou_noise import OUNoise
from replay_buffer import ReplayBuffer


class FirstAgent(base_agent.BaseAgent):

    def __init__(self):
        super(FirstAgent, self).__init__()

        # actor models
        self.actor_local = None
        self.actor_target = None

        # critic models
        self.critic_local = None
        self.critic_target = None

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01  # for soft update of target parameters

    def step(self, obs):
        super(FirstAgent, self).step(obs)

        # TODO: implement action choosing stuff
        return actions.FUNCTIONS.no_op()

    def learn(self, experiences):
        # TODO: implement learning function
        pass

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
