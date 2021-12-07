import torch.nn as nn
import torch
from algorithms.utils.math import normal_log_density


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(128, 128), log_std=0):
        """Constructor of Policy network

        Args:
            state_dim: state dimension
            action_dim: action dimension
            hidden_size: hidden layers' sizes. Defaults to (128, 128).
            log_std: log standard deviation. Defaults to 0.
        """
        super().__init__()
        self.is_disc_action = False
        self.activation = torch.tanh

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        """Forward pass of Policy network

        Args:
            x: input

        Returns:
            action mean, action log standard deviation, action standard deviation
        """
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
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
        return action

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
