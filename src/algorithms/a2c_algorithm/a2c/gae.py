import torch
from algorithms.utils import to_device

# Explanation: https://avandekleut.github.io/a2c/
# Reference: https://slm-lab.gitbook.io/slm-lab/development/algorithms/a2c
# Reference: https://github.com/kengz/SLM-Lab


def generalized_advantage_estimation(rewards, masks, values, gamma, tau, device):
    """Estimates advantages values for A2C with GAE

    Args:
        rewards: rewards of current state
        masks: masks
        values: current values
        gamma: discount factor
        tau: tau
        device: cpu

    Returns:
        Calculated advantages and values + advantages
    """
    rewards, masks, values = to_device(torch.device("cpu"), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0

    # for t in reversed(range(T = len(rewards)))
    for i in reversed(range(rewards.size(0))):

        # delta = rewards[t] + gamma * v_preds[t + 1] * not_dones[t] - v_preds[t]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]

        # gae[t] = future_gae = delta + gamma * tau * not_done[t] * future_gae
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages  # v_targets = advs + v_preds
    advantages = (
        advantages - advantages.mean()
    ) / advantages.std()  # standardize only for advs, not v_targets

    advantages, returns = to_device(
        device, advantages, returns
    )  # trained with gpu or cpu

    return advantages, returns
