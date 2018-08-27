"""Test agent to solve the move to beacon environment"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pysc2.lib import actions, features
from pysc2.env import environment, available_actions_printer

from agents.actor import Actor
from agents.critic import Critic

"""
it seems like this agent thing is kind of similar to the task script from
the quadcopter project.
"""


class Agent(object):

    def __init__(self):
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.w = None
        self.action_space = 0
        self.observation_space = 0

        self.local_actor = None
        self.target_actor = None
        self.local_critic = None
        self.target_critic = None

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

        # can't do this
        self.action_space = actions.ActionSpace
        print('setting up agent with action space:', self.action_space)

        self.w = np.random.normal(
            size=()
        )

        # number of different actions is pretty large: 541 different action id's
        self.action_space = len(self.action_spec.functions._func_list)

        # lets try to make the observation_space the feature_screen as a start
        # maybe not all of them. dims are (17,84,84)
        self.observation_space = self.obs_spec.feature_screen

        #observation_space = obs_spec.player_relative
        print('this is the obs_space for player_relative:', self.observation_space)

        print('made it!')
        self.local_actor = Actor(
            state_size=self.observation_space,
            action_size=self.action_space,
            action_low=0, action_high=541)


    def reset(self):
        self.episodes += 1

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward

        # not all actions are available at every time step
        available_actions = obs.observation.available_actions

        # print(available_actions)

        # take a random action
        function_id = np.random.choice(obs.observation.available_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)

        # return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
