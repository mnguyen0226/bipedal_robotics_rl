# Implementation of Bipedal Walking with PPO + GAE - Proximal Policy Optimization with Generalized Advantage Estimation
# By Minh Nguyen
# ECE 5984 - Reinforcement Learning
# 11/21/2021

# Inspired Reference: https://github.com/nikhilbarhate99/PPO-PyTorch
# Inspired Reference: https://github.com/4kasha/CartPole_PPO
# Reading: https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

import gym
import torch
import os
import sys
import pickle
import time
import math
from os import path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.utils.additional_torch import to_device
from algorithms.ppo_algorithm import Policy
from algorithms.ppo_algorithm import Value
from algorithms.ppo_algorithm import ppo_step
from algorithms.a2c_algorithm import generalized_advantage_estimation
from algorithms.a2c_algorithm import BipedalWalkerAgent
from algorithms.utils import AlgorithmRunManagement
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# initialize global variable
L2_REG = 1e-3
GAMMA = 0.99
MAX_NUM_ITER = 1000  # 50000
RENDER = False
MIN_BATCH_SIZE = 2048
LOG_INTERVAL = 1
SAVE_MODEL_INTERVAL = 100
CLIP_EPSILON = 0.2
ENV_NAME = "BipedalWalker-v2"
OPTIM_EPOCHS = 10
OPTIM_BATCH_SIZE = 64

# set datatype
dtype = torch.float64
torch.set_default_dtype(dtype)

# set device
device = (
    torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")
)
if torch.cuda.is_available():
    torch.cuda.set_device(0)

# environment
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
running_state = AlgorithmRunManagement((state_dim,), clip=5)

# seeding
np.random.seed(1)
torch.manual_seed(1)
env.seed(1)

# define actor and critic network
policy_net = Policy(state_dim, env.action_space.shape[0], log_std=-1.0)
value_net = Value(state_dim)

# test trained model
# policy_net, value_net, running_state = pickle.load(open("assets/learned_models/ppo_algorithm/Bipedal_walker_v2_ppo.p", "rb"))

# comment out these two line below if you decided to test the train model line above
policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=4e-4)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=8e-4)

# create agent
agent = BipedalWalkerAgent(
    env,
    policy_net,
    device,
    running_state=running_state,
    render=False,
    num_threads=1,
)


def saved_assets_dir():
    """Saves trained model in assets directory

    Returns:
        Paths to asset directory
    """
    return path.abspath(
        path.join(path.dirname(path.abspath(__file__)), "../../../assets")
    )


def update_ppo_params(batch, tau):
    """Updates training parameters by taking steps from PPO algorithm

    Args:
        batch: input batch
    """
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    (
        advantages,
        returns,
    ) = generalized_advantage_estimation(  # get estimated advantage from stepping trajectories
        rewards=rewards, masks=masks, values=values, gamma=GAMMA, tau=tau, device=device
    )

    # mini batch PPO update
    optim_iter_num = int(math.ceil(states.shape[0] / OPTIM_BATCH_SIZE))

    for _ in range(OPTIM_EPOCHS):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = (
            states[perm].clone(),
            actions[perm].clone(),
            returns[perm].clone(),
            advantages[perm].clone(),
            fixed_log_probs[perm].clone(),
        )

        for i in range(optim_iter_num):
            ind = slice(
                i * OPTIM_BATCH_SIZE, min((i + 1) * OPTIM_BATCH_SIZE, states.shape[0])
            )
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = (
                states[ind],
                actions[ind],
                advantages[ind],
                returns[ind],
                fixed_log_probs[ind],
            )

            ppo_step(  # run PPO algorithm updates
                policy_net=policy_net,
                value_net=value_net,
                optimizer_policy=optimizer_policy,
                optimizer_value=optimizer_value,
                optim_value_iter_num=1,
                states=states_b,
                actions=actions_b,
                returns=returns_b,
                advantages=advantages_b,
                fixed_log_probs=fixed_log_probs_b,
                clip_epsilon=CLIP_EPSILON,
                l2_reg=L2_REG,
            )


def ppo_main():
    """User Interface"""
    # log training date
    localtime = time.asctime(time.localtime(time.time()))
    with open("assets/training_times/ppo_algorithm/training_time.txt", "a") as f:
        f.write(localtime)
        f.write("----------\n")

    # list of tau / lambda value
    # tau_list = [0.50, 0.70, 0.90, 0.95, 0.97, 0.99]
    tau_list = [0.99]
    color_list = ["black", "red", "yellow", "green", "darkblue", "orange"]

    # plot
    plot = plt.figure()
    subplot = plot.add_subplot()

    for i in range(len(tau_list)):
        t0 = time.time()  # for logging training time

        # plot
        xval, yval = [], []
        plt.xlabel("Number Episodes")
        plt.ylabel("Rewards")
        plt.title(
            "Bipedal Walker v2 with PPO_GAE\ngamma=0.90, num_episodes=1000, clip_epsilon=0.2\nL2_reg=1e-3, min_batch_size=2048, optim_batch_size=64"
        )
        string_label = "λ/tau = " + str(tau_list[i])
        (plotLine,) = subplot.plot(xval, yval, color_list[i], label=string_label)
        subplot.set_xlim([0, MAX_NUM_ITER])
        subplot.set_ylim([-300, 400])

        # plot legend
        plot.legend(loc="upper right")

        # run iteration
        for i_iter in range(MAX_NUM_ITER):
            # generates multiple trajectories that reach the min_batch_size
            batch, log = agent.collect_samples(MIN_BATCH_SIZE, RENDER)

            # update the parameter for each time step - second for loop
            update_ppo_params(batch, tau_list[i])

            if i_iter % LOG_INTERVAL == 0:
                print(
                    f'Episode {i_iter+1} finished. Highest reward: {log["max_reward"]}'
                )

            # if agent was able to get the highest rewards of 300, then log the episode number
            if log["max_reward"] >= 300.0:
                with open(
                    "assets/log_episodes_300_rewards/ppo_algorithm/log_300.txt", "a"
                ) as f:
                    f.write(
                        "Episode "
                        + str(i_iter + 1)
                        + " of lambda/tau "
                        + str(tau_list[i])
                        + " has the reward of "
                        + str(log["max_reward"])
                        + ".\n"
                    )
                    f.close()

            # plot
            xval.append(i_iter)
            yval.append(log["max_reward"])
            plotLine.set_xdata(xval)
            plotLine.set_ydata(yval)
            plot.savefig("./results/ppo_max_reward")

            if SAVE_MODEL_INTERVAL > 0 and (i_iter + 1) % SAVE_MODEL_INTERVAL == 0:
                to_device(torch.device("cpu"), policy_net, value_net)

                pickle.dump(  # write the trained model to folder
                    (policy_net, value_net, running_state),
                    open(
                        os.path.join(
                            saved_assets_dir(),
                            "learned_models/ppo_algorithm/cpu_bipedal_walker_v2_ppo.p",
                        ),
                        "wb",
                    ),
                )
                to_device(device, policy_net, value_net)

            # clean up gpu memory after every iteration
            torch.cuda.empty_cache()

        t1 = time.time()
        print(f"All episodes finished. Training time of PPO is: {t1-t0}")

        # write training time to file
        with open("assets/training_times/ppo_algorithm/training_time.txt", "a") as f:
            f.write(
                "- The training time for 1000 episode of PPO_GAE with the lambda-return/tau-return of "
            )
            f.write(str(tau_list[i]))
            f.write(" is: ")
            f.write(str(t1 - t0))
            f.write(" seconds.\n")
            f.close()
