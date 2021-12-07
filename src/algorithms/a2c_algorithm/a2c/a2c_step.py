# Reference: https://github.com/grantsrb/Pytorch-A2C

import torch


def a2c_step(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    states,
    actions,
    returns,
    advantages,
    l2_reg,
):
    """Updates Critic network and Policy network parameter with first order optimization

    Args:
        policy_net: Policy network
        value_net: Critic value network
        optimizer_policy: optimizer or policy network - Adam
        optimizer_value: optimizer of critic network - Adam
        states: states array
        actions: action array
        returns: returns values
        advantages: estimated advantage values
        l2_reg: L2 Regularization
    """
    # update Critic value network
    values_pred = value_net(states)
    # calculate value loss with MeanSquaredError
    value_loss = (values_pred - returns).pow(2).mean()

    # weight decays with L2 Regularization
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg

    optimizer_value.zero_grad()  # initialize gradients to 0s
    value_loss.backward()  # update Critic parameters with Adam optimize with back propagation
    optimizer_value.step()  # update gradients

    # update Policy network
    log_probs = policy_net.get_log_prob(states, actions)  # get log probabilities

    # calculate policy loss
    policy_loss = -(log_probs * advantages).mean()
    optimizer_policy.zero_grad()  # initialize gradients to 0s
    policy_loss.backward()  # u pdate Actor parameters with Adam optimizer with back propagation
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(), 40
    )  # clip the gradient to avoid overfit of under fit
    optimizer_policy.step()  # update gradients
