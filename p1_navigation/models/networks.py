import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkFC(nn.Module):
    """Full-connected network template."""

    def __init__(self, state_size, action_size, fc1_units, fc2_units, seed=42):
        """Initialize parameters and build model.

        Keywords:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            seed (int): random seed
        """
        super(QNetworkFC, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
