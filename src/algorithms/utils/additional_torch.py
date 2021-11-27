import torch
import numpy as np
from torch.autograd.variable import Variable


def to_device(device, *args):
    """Gets the model to train on CPU or GPU

    Args:
        device: GPU or CPU

    Returns:
        Have all training parameters to be trained on CPU or GPU
    """
    return [x.to(device) for x in args]


def to_np(x):
    """Converts from Tensor datatype to numpy datatype

    Args:
        x: Tensor datatype

    Returns:
        Numpy datatype
    """
    return x.data.cpu().numpy()


def to_var(x):
    """Converts from Tensor datatype to Variable datatype

    Args:
        x: Tensor datatype

    Returns:
        Variable datatype
    """
    if torch.cuda.is_available():
        x = x.cuda()
    var = Variable(x)
    return var
