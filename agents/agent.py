# import the other relevant sample code
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import random
from collections import namedtuple, deque
import copy

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=32, kernel_initializer='random_uniform')(states)
        act1 = LeakyReLU(alpha=0.2)
        net = act1(net)
        net = layers.Dropout(0.15)(net)
        net = layers.Dense(units=64, kernel_initializer='random_uniform')(net)
        act2 = LeakyReLU(alpha=0.2)
        net = act2(net)
        net = layers.Dropout(0.30)(net)
        net = layers.Dense(units=32, kernel_initializer='random_uniform')(net)
        act3 = LeakyReLU(alpha=0.2)
        net = act3(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        #actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
        #    name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=raw_actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * raw_actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, kernel_initializer='random_uniform')(states)
        act1 = LeakyReLU(alpha=0.2)
        net_states = act1(net_states)
        net_states = layers.Dense(units=64, kernel_initializer='random_uniform')(net_states)
        act2 = LeakyReLU(alpha=0.2)
        net_states = act2(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, kernel_initializer='random_uniform')(actions)
        act1 = LeakyReLU(alpha=0.2)
        net_actions = act1(net_actions)
        net_actions = layers.Dense(units=64, kernel_initializer='random_uniform')(net_actions)
        act2 = LeakyReLU(alpha=0.2)
        net_actions = act2(net_actions)

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
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 5.0
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.01  # for soft update of target parameters

        # Track reward/score stats
        self.episode_reward = 0.0
        self.step_count = 0
        self.best_score = -np.inf
        self.score = 0.0

        self.movAvgRecord = [0]
        self.movAvgIndex = 0
        self.currentMovAvg = 0.
        self.bestMovAvg = -np.inf
        self.rewardBuf = []

        self.epsBuf_time = []
        self.epsBuf_pose = []
        self.epsBuf_v = []
        self.epsBuf_angular_v = []
        self.epsBuf_rotorSpeed = []
        self.epsBuf_score = []

    def reset_episode(self):
        self.movAvgIndex += 1
        numEpisodeAvg = 25
        if self.movAvgIndex >= numEpisodeAvg:
            index = self.movAvgIndex - numEpisodeAvg + 1
            self.movAvgRecord.append(self.movAvgRecord[index-1] + (1 / numEpisodeAvg) * (self.score - self.movAvgRecord[index-1]))
            self.currentMovAvg = self.movAvgRecord[index]

        else:
            self.movAvgRecord[0] = self.movAvgRecord[0] + (1 / self.movAvgIndex) * (self.score - self.movAvgRecord[0])
            self.currentMovAvg = self.movAvgRecord[0]

        if self.currentMovAvg > self.bestMovAvg:
            self.bestMovAvg = self.currentMovAvg   
        self.rewardBuf.append(self.score)

        self.episode_reward = 0.0
        self.step_count = 0
        self.score = 0.0
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state

        self.epsBuf_time = []
        self.epsBuf_pose = []
        self.epsBuf_v = []
        self.epsBuf_angular_v = []
        self.epsBuf_rotorSpeed = []
        self.epsBuf_score = []

        return state

    def step(self, action, reward, next_state, done):
        self.episode_reward += reward
        self.step_count += 1

         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

        self.epsBuf_time.append(self.task.sim.time)
        self.epsBuf_pose.append(self.task.sim.pose)
        self.epsBuf_v.append(self.task.sim.v)
        self.epsBuf_angular_v.append(self.task.sim.angular_v)
        self.epsBuf_score.append(self.score)

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        self.epsBuf_rotorSpeed.append(list(action))
        action = (action * self.action_range) + self.action_low
        output = list(action + self.noise.sample())
        return output  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        self.score = self.episode_reward / float(self.step_count) if self.step_count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)