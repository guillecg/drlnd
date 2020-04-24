import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from p1_navigation.models.networks import QNetworkFC
from p1_navigation.models.utils import ReplayBuffer


class AgentDQN(object):
    """Agent class that handles the training of the networks and provides
    outputs as actions

    Keywords:
        state_size (int): dimension of each state
        action_size (int): dimension of each action
        hyperparams (dict): hyperparameters for training the agent
        device (device): cuda or cpu to process tensors
        seed (int): random seed

    """

    def __init__(self, action_size, state_size, hyperparams, device, seed=42):
        self.seed = seed
        self.device = device

        # Environment parameters definition
        self.action_size = action_size
        self.state_size = state_size

        # Hyperparameters definition
        self.batch_size = 64         # minibatch size
        self.buffer_size = int(1e5)  # replay buffer size
        self.learn_every = 2         # how often to learn
        self.gamma = 0.99            # discount factor
        self.tau = 1e-3              # for soft update of target parameters
        self.lr = 5e-4               # learning rate
        self.epsilon = 0.05          # for e-greedy policy

        # Update defined hyperparameters
        self.__dict__.update(hyperparams)

        # Q-networks
        fc1_units = int(self.state_size * 0.75)
        fc2_units = int(self.state_size * 0.50)
        self.qnetwork_local = QNetworkFC(
            state_size=self.state_size,
            action_size=self.action_size,
            fc1_units=fc1_units,
            fc2_units=fc2_units,
            seed=self.seed
        )
        self.qnetwork_target = QNetworkFC(
            state_size=self.state_size,
            action_size=self.action_size,
            fc1_units=fc1_units,
            fc2_units=fc2_units,
            seed=self.seed
        )

        self.qnetwork_local.to(self.device)
        self.qnetwork_target.to(self.device)

        self.optimizer = optim.Adam(
            params=self.qnetwork_local.parameters(),
            lr=self.lr
        )

        # Replay memory
        self.memory = ReplayBuffer(
            action_size=self.action_size,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed
        )

        # Initialize time step (for updating every self.update_every steps)
        self.t_step = 0

    @staticmethod
    def policy(values, epsilon):
        """Applies an epsilon greedy policy to action values."""

        if np.random.rand() <= epsilon:
            action = np.random.randint(low=0, high=len(values) - 1)
        else:
            action = np.argmax(values.cpu().data.numpy())

        return action

    def select_action(self, state, epsilon=None):
        """Choose actions for given state as per current policy.

        Keywords:
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection

        Returns:
            action (float): chosen action

        """
        epsilon = epsilon or self.epsilon

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Avoid training when acting == retrieving action values
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        action_values = action_values.view(-1)
        action = self.policy(values=action_values, epsilon=epsilon)

        return action

    def step(self, state, action, reward, next_state, done):
        """Saves experience into replay memory and learns from it if enough
        experiences have been collected.

        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every learn_every time steps.
        self.t_step = (self.t_step + 1) % self.learn_every

        if self.t_step == 0:
            # If enough samples are available in memory,
            # get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def learn(self, experiences, gamma=None):
        """Update value parameters using given batch of experience tuples.

        Keywords:
            experiences (Tuple[torch.Tensor]): ((s, a, r, s', done))
            gamma (float): discount factor

        """
        gamma = gamma or self.gamma

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.qnetwork_target(next_states).detach()
        q_targets_next = q_targets_next.max(1)[0].unsqueeze(1)

        # Compute Q targets for current states (reward + discounted)
        # NOTE: (1 - dones) avoids taking into account terminal states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        # NOTE: gather here takes selected actions (int) across dim=1 (columns)
        # NOTE 2: actions here are the e-greedy selected actions
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(
            local_model=self.qnetwork_local,
            target_model=self.qnetwork_target,
            tau=self.tau
        )

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Keywords:
            local_model (PyTorch model): from which weights will be copied
            target_model (PyTorch model): to which weights will be copied
            tau (float): interpolation parameter
        """
        iterable = zip(target_model.parameters(), local_model.parameters())

        for target_param, local_param in iterable:
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
