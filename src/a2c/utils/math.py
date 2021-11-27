import torch
import math
import torch.nn as nn


def normal_entropy(std):
    """Returns normalized entropy valies

    Args:
        std: standard deviation

    Returns:
        normalized entropy
    """
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    """Returns normalized log density

    Args:
        x: input
        mean: mean
        log_std: log standard deviation
        std: standard deviation

    Returns:
        normalized log density
    """
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    nld = log_density.sum(1, keepdim=True)
    return nld


def set_init(layers):
    """Initializes values of layers in neural networks

    Args:
        layers: array of layers
    """
    for l in layers:
        nn.init.normal(l.weight, mean=0.0, std=0.1)
        nn.init.constant(l.bias, 0.0)
