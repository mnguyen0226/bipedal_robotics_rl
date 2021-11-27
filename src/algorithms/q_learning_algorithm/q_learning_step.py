# Implementation of Bipedal Walking with Q Learning
# By Minh Nguyen
# ECE 5984 - Reinforcement Learning
# 11/20/2021

import numpy as np
import random
import math


def update_q_table(q_table, state, action, reward, alpha, gamma, next_state=None):
    """Updates Q table

    Args:
        q_table: current Q table
        state: current state
        action: current action space
        reward: reward of the current state, action
        alpha: learning rate
        gamma: discount factor
        next_state: next state. Defaults to None.

    Returns:
        new Q value
    """
    curr_q_val = q_table[state][action]  # Q(s,a)
    if next_state is not None:  # Q'(s,a)
        next_q_val = np.max(q_table[next_state])
    else:
        next_q_val = 0
    target_q_val = reward + (gamma * next_q_val)  # [R + gamma*max_a(Q'(s,a) - Q(s,a))]

    # Q(s,a) <- Q(s,a) + alpha[R + gamma*max_a(Q'(s,a) - Q(s,a))]
    new_q_val = curr_q_val + (alpha * (target_q_val - curr_q_val))
    return new_q_val


def get_next_action(q_table, epsilon, state):
    """Returns the actions depends on the epsilon value for epsilon-greedy strategy

    Args:
        q_table: current Q table
        epsilon: epsilon value
        state: current state space

    Returns:
        explorative or exploitative action.
    """
    if random.random() < epsilon:  # explore the environment by taking random action
        action = ()
        for _ in range(0, 4):
            action += (random.randint(0, 9),)
    else:  # exploit the environment by taking the ation that returns the highest rewards
        action = np.unravel_index(np.argmax(q_table[state]), q_table[state].shape)

    return action


def discretize_state(state):
    """Discretizes continuous state space to discrete state space

    Args:
        state: current continuous state space

    Returns:
        discretized state space
    """

    state_bounds = [
        (0, math.pi),
        (-2, 2),
        (-1, 1),
        (-1, 1),
        (0, math.pi),
        (-2, 2),
        (0, math.pi),
        (-2, 2),
        (0, 1),
        (0, math.pi),
        (-2, 2),
        (0, math.pi),
        (-2, 2),
        (0, 1),
    ]

    discrete_state = []
    for i in range(len(state)):
        idx = int(
            (state[i] - state_bounds[i][0])
            / (state_bounds[i][1] - state_bounds[i][0])
            * 19
        )
        discrete_state.append(idx)
    return tuple(discrete_state)


def convert_next_action(next_action):
    """Converts get next action to next action

    Args:
        next_action: get next action from epsilon greedy policy

    Returns:
        next action
    """
    action = []
    for i in range(len(next_action)):
        next_val = next_action[i] / 9 * 2 - 1
        action.append(next_val)

    return tuple(action)


def q_learning(env, num_episode, q_table, highest_reward, alpha, gamma, render=False):
    """Implement Q Learning

    Args:
        env: Bipedal Walking environment
        num_episode: number of episodes
        q_table: initial Q table
        highest_reward: current highest reward
        alpha: learning rate
        gamma: discount factor
        render (bool, optional): render environment. Defaults to False.

    Returns:
        total reward, highest reward
    """
    if render:
        env.render()
    state = discretize_state(env.reset()[0:14])
    total_reward = 0
    epsilon = 1.0 / num_episode * 0.004

    while True:
        next_action = convert_next_action(
            get_next_action(q_table=q_table, epsilon=epsilon, state=state)
        )
        discretized_next_action = get_next_action(
            q_table=q_table, epsilon=epsilon, state=state
        )
        next_state, reward, done, info = env.step(next_action)
        next_state = discretize_state(next_state[0:14])
        total_reward += reward
        q_table[state][discretized_next_action] = update_q_table(
            q_table=q_table,
            state=state,
            action=discretized_next_action,
            alpha=alpha,
            reward=reward,
            gamma=gamma,
            next_state=next_state,
        )
        state = next_state
        if done:
            break

    if total_reward > highest_reward:
        highest_reward = total_reward
    return total_reward, highest_reward
