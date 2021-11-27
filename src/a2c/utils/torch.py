import torch
import numpy as np


def to_device(device, *args):
    """Gets the model to train on CPU or GPU

    Args:
        device: GPU or CPU

    Returns:
        Have all training parameters to be trained on CPU or GPU
    """
    return [x.to(device) for x in args]


def get_flat_params_from(model):
    """Haves all training parameters to be in 1 dimension tensor

    Args:
        model: model

    Returns:
        flatten paramters
    """
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    """Sets flat parameters

    Args:
        model: model
        flat_params: parameters from get_flat_params_from()
    """
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind : prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size


def get_flat_grad_from(inputs, grad_grad=False):
    """Returns gradients values in 1 dimension

    Args:
        inputs: input gradients
        grad_grad: Defaults to False.

    Returns:
        Flatten gradients array
    """
    grads_arr = []
    for param in inputs:
        if grad_grad:
            grads_arr.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:  # if there is no gradient in the parameter
                grads_arr.append(torch.zeros(param.view(-1).shape))
            else:
                grads_arr.append(param.grad.view(-1))

    flat_grad = torch.cat(grads_arr)
    return flat_grad


def compute_flat_grad(
    output, inputs, filter_input_ids=set(), retain_graph=False, create_graph=False
):
    """Returns flatten gradients through filter

    Args:
        output: outputs of the differentiated function.
        inputs: inputs w.r.t. which the gradient will be returned (and not accumulated into .grad).
        filter_input_ids: filter input dimension. Defaults to set().
        retain_graph: if False, the graph used to compute the grad will be freed. Note that in nearly all cases setting this 
            option to True is not needed and often can be worked around in a much more efficient way. Defaults to the value 
            of create_graph.
        create_graph: if True, graph of the derivative will be constructed, allowing to compute higher order derivative 
            products. Default: False.

    Returns:
        Filtered gradients
    """
    if create_graph:
        retain_graph = True

    inputs = list(inputs) # convert to list
    params = []
    for i, param in enumerate(inputs):
        if i not in filter_input_ids:
            params.append(param)

    grads = torch.autograd.grad( # get gradient list
        output, params, retain_graph=retain_graph, create_graph=create_graph
    )

    idx = 0
    out_grads = []
    for i, param in enumerate(inputs):
        if i in filter_input_ids:
            out_grads.append(
                torch.zeros(
                    param.view(-1).shape, device=param.device, dtype=param.dtype
                )
            )
        else:
            out_grads.append(grads[idx].view(-1))
            idx += 1
    grads = torch.cat(out_grads)

    for param in params:
        param.grad = None
        
    return grads
