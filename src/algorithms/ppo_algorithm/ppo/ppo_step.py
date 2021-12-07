import torch


def ppo_step(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    optim_value_iter_num,
    states,
    actions,
    returns,
    advantages,
    fixed_log_probs,
    clip_epsilon,
    l2_reg,
):
    """Updates Critic network and Policy network with first order optimization

    Args:
        policy_net: Policy network
        value_net: Critic value network
        optimizer_policy: optimizer or policy network - Adam
        optimizer_value: optimizer or critic network - Adam
        optim_value_iter_num: optimizer value iteration number
        states: states array
        actions: action array
        returns: returns values
        advantages: estimated advantage values
        fixed_log_probs: fixed log probabilities
        clip_epsilon: clip epsilon to avoid overfit or underfit
        l2_reg: L2 Regularization
    """
    # update Critic value network
    for _ in range(optim_value_iter_num):
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean() # MSE for critic network
        # weight decays with L2 Regularization
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg

        optimizer_value.zero_grad()  # initialize gradients to 0s
        
        # update Critic parameters with Adam optimizer using back propagation
        value_loss.backward()  
        optimizer_value.step() 

    # update Policy network
    log_probs = policy_net.get_log_prob(states, actions)  # get log probabilities
    
    # calculate the clipped surrogate objective function 
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages 
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean() # policy net loss
    
    optimizer_policy.zero_grad()  # initialize gradients to 0s
    
    # update Actor parameters with Adam optimizer using back propagation
    policy_surr.backward() 
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(), 40
    )  # clip the gradient to avoid overfit of under fit
    optimizer_policy.step()  # update gradients
