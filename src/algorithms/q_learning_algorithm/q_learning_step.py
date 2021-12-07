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


def discretize_state(state):
    """Discretizes continuous state space to discrete state space with 14 physical values

    Args:
        state: current continuous state space

    Returns:
        discretized state space
    """

    # discretize observation space
    # https://github.com/openai/gym/wiki/BipedalWalker-v2
    obs_state_bounds = [
        (0, math.pi),  # hull_angle
        (-2, 2),  # hull_angular_velocity
        (-1, 1),  # vel_x
        (-1, 1),  # vel_y
        (0, math.pi),  # hip_joint_1_angle
        (-2, 2),  # hip_joint_1_speed
        (0, math.pi),  # knee_joint_1_angle
        (-2, 2),  # knee_joint_1_speed
        (0, 1),  # leg_1_ground_contact_flag
        (0, math.pi),  # hip_joint_2_angle
        (-2, 2),  # hip_joint_2_speed
        (0, math.pi),  # knee_joint_2_angle
        (-2, 2),  # knee_joint_2_speed
        (0, 1),  # leg_2_ground_contact_flag
    ]

    # create an empty obs_discrete_state array to store converted discrete state array
    obs_discrete_state = []

    for i in range(len(state)):
        converted_i = int(
            (state[i] - obs_state_bounds[i][0])
            / (obs_state_bounds[i][1] - obs_state_bounds[i][0])
            * 19  # 19 is arbitrary integer
        )
        obs_discrete_state.append(converted_i)

    ds = tuple(
        obs_discrete_state
    )  # convert collected discrete state array into tuple to maintain same shape
    return ds


# Epsilon Greedy Action: https://deeplizard.com/learn/video/mo96Nqlo1L8
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


def convert_next_action(next_action):
    """Converts get next action to next action

    Args:
        next_action: get next action from epsilon greedy policy

    Returns:
        next action
    """
    action = []
    for i in range(len(next_action)):
        next_val = ((next_action[i] / 9) * 2) - 1
        action.append(next_val)

    ta = tuple(action)  # convert action array into tuple to maintain same shape
    return ta


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
    if render:  # set rendering state
        env.render()

    # initialize state, reward, epsilon greedy rate
    state = discretize_state(
        env.reset()[0:14]
    )  # get 14 values without the lidar measurement
    total_reward = 0  # initialize total reward
    epsilon = (
        1.0 / num_episode * 0.004
    )  # initialize epsilon greedy that will decrease (exploit) as the num_episode increase

    while True:
        next_action = convert_next_action(  # choose next action
            get_next_action(q_table=q_table, epsilon=epsilon, state=state)
        )

        discretized_next_action = get_next_action(  # discretize the next action
            q_table=q_table, epsilon=epsilon, state=state
        )

        # agent take action and collect info (continuous)
        next_state, reward, done, _ = env.step(next_action)

        # discretize the next state
        next_state = discretize_state(next_state[0:14])

        # collect reward
        total_reward += reward

        # update q table
        q_table[state][discretized_next_action] = update_q_table(
            q_table=q_table,
            state=state,
            action=discretized_next_action,
            alpha=alpha,
            reward=reward,
            gamma=gamma,
            next_state=next_state,
        )

        # set current state to next state
        state = next_state

        if done:
            break

    # return the highest reward in the episode
    if total_reward > highest_reward:
        highest_reward = total_reward

    return total_reward, highest_reward
