import numpy as np
import random

from pysc2.agents import base_agent
from pysc2.lib import actions, units

from keras import layers, models, optimizers
from keras import backend as K

from scripted_agents import agent_utils as utils


class MoveToBeaconAgent(base_agent.BaseAgent):

    def __init__(self):
        super(MoveToBeaconAgent, self).__init__()
        self.score = None

        state_size = (64, 64)

        actor = Actor()
        critic = Critic()

    def step(self, obs):
        super(MoveToBeaconAgent, self).step(obs)

        x = random.randint(0, 63)
        y = random.randint(0, 63)
        if utils.can_do(obs, actions.FUNCTIONS.Move_minimap.id):
            return actions.FUNCTIONS.Move_minimap('now', (x, y))

        marine_units = utils.get_units_by_type(obs, units.Terran.Marine)
        marine_unit = random.choice(marine_units)

        return actions.FUNCTIONS.select_point('select_all_type', (marine_unit.x, marine_unit.y))


class Actor:

    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        self.__build_model()

    def __build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=64, activation='relu')(states)
        net = layers.Dropout(rate=8)(net)
        net = layers.Dense(units=256, activation='relu')(net)
        net = layers.Dropout(rate=32)(net)
        net = layers.Dense(units=128, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
                                   name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                                name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0002)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.__build_model()

    def __build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        net_states = layers.Dense(units=256, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        net_actions = layers.Dense(units=256, activation='relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.00003)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivati    ve of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
