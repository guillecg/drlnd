import random

import numpy as np

import torch

from collections import namedtuple, deque

import copy


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, device, seed=42):
        """Initialize a ReplayBuffer object.

        Keywords:
            buffer_size (int): maximum size of buffer
            device (device): cuda or cpu to process tensors
            seed (int): random seed

        """
        self.seed = random.seed(seed)
        self.device = device

        # Create memory object
        self.memory = deque(maxlen=buffer_size)

        # Create experience object
        self.experience = namedtuple(
            typename='Experience',
            field_names=['state', 'action', 'reward', 'next_state', 'done']
        )

    def add(self, data):
        """Add a new experience to memory."""
        state, action, reward, next_state, done = data
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=batch_size)

        states = torch\
            .from_numpy(
                np.vstack([e.state for e in experiences if e is not None])
            )\
            .float()\
            .to(self.device)

        actions = torch\
            .from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )\
            .float()\
            .to(self.device)

        rewards = torch\
            .from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )\
            .float()\
            .to(self.device)

        next_states = torch\
            .from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )\
            .float()\
            .to(self.device)

        dones = torch\
            .from_numpy(
                np.vstack([e.done for e in experiences if e is not None])
                .astype(np.uint8)
            )\
            .float()\
            .to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class ReplayBufferOther(object):
    """Buffer to store tuples of experience replay"""

    def __init__(self, max_size=1000000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """

        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """Add experience tuples to buffer

        Args:
            data (tuple): experience replay tuple
        """

        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size

        Args:
            batch_size (int): size of sample
        """

        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in ind:
            s, a, r, s_, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(s_, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states),\
               np.array(actions),\
               np.array(rewards).reshape(-1, 1),\
               np.array(next_states),\
               np.array(dones).reshape(-1, 1)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.storage)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.seed = random.seed(seed)

        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state

        dx = \
            self.theta * (self.mu - x) + \
            self.sigma * np.random.standard_normal(self.size)

        self.state = x + dx

        return self.state
