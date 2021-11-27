import torch.nn as nn
import torch.nn.functional as F
from utils import set_init


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(200, 128)):
        """Constructor of Value Critic network

        Args:
            state_dim: state dimension
            hidden_size: hidden layers' sizes. Defaults to (200,128).
        """
        super().__init__()
        self.activateion = F.relu
        self.affine_layers_v = nn.ModuleList()
        self.bn_layers_v = nn.ModuleList()
        last_dim = state_dim

        for hs in hidden_size:
            self.affine_layers_v.append(nn.Linear(last_dim, hs))
            self.bn_layers_v.append(nn.BatchNorm1d(hs, momentum=0.5))
            last_dim = hs

        self.value_head = nn.Linear(last_dim, 1)
        set_init(self.affine_layers_v)
        set_init([self.value_head])

    def forward(self, x):
        """Forward pass of Value Critic network

        Args:
            x: input

        Returns:
            Output value
        """
        self.eval()

        for affine, bn in zip(self.affine_layers_v, self.bn_layers_v):
            x = affine(x)
            x = bn(x)
            x = self.activation(x)

        out = self.value_head(x)

        return out
