import math
import torch.nn as nn

# Reference: https://medium.com/@mansiarora_20448/probability-distributions-in-machine-learning-98811eb1e8ea

def normal_log_density(x, mean, log_std, std):
    """Returns normalized log density, aka normal distribution, aka Gaussian distribution

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
    Set network weight with means and standard deviation
    Set network bias

    Args:
        layers: array of layers
    """
    for l in layers:
        nn.init.normal(l.weight, mean=0.0, std=0.1)
        nn.init.constant(l.bias, 0.0)
