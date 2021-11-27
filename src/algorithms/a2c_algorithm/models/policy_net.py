import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from algorithms.utils import normal_log_density
from algorithms.utils import set_init


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(200, 128), log_std=0):
        """Constructor of Policy network

        Args:
            state_dim: state dimension
            action_dim: action dimension
            hidden_size: hidden layers' sizes . Defaults to (200, 128).
            log_std (int, optional): log standard deviation. Defaults to 0.
        """
        super().__init__()
        self.is_disc_action = False
        self.activation = F.relu
        self.affine_layers_p = nn.ModuleList()
        self.bn_layers_p = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers_p.append(nn.Linear(last_dim, nh))
            self.bn_layers_p.append(nn.BatchNorm1d(nh, momentum=0.5))
            last_dim = nh
        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
        self.entropy_coef = 0.01

        set_init(self.affine_layers_p)
        set_init([self.action_mean])

    def forward(self, x):
        """Forward pass of Policy network

        Args:
            x: input

        Returns:
            action mean, action log standard deviation, action standard deviation
        """
        self.eval()
        for affine, bn in zip(self.affine_layers_p, self.bn_layers_p):
            x = affine(x)
            x = bn(x)
            x = self.activation(x)

        action_mean = F.tanh(self.action_mean(x))
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        """Selects the action that gave the highest reward

        Args:
            x: input

        Returns:
            Selected action value
        """
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        selected_action = action.data

        return selected_action

    def get_log_prob(self, x, actions):
        """Returns log probability

        Args:
            x: input
            actions: actions

        Returns:
            Normalized log density action
        """
        action_mean, action_log_std, action_std = self.forward(x)
        normalized_log_action = normal_log_density(
            actions, action_mean, action_log_std, action_std
        )

        return normalized_log_action
