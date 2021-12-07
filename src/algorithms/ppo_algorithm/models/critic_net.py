import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128)):
        """Constructor of Critic network

        Args:
            state_dim: state dimension
            hidden_size: hidden layers' sizes. Defaults to (128, 128).
        """
        super().__init__()
        self.activation = torch.tanh
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        """Forward pass of Value Critic network

        Args:
            x: input

        Returns:
            Output value
        """
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value
