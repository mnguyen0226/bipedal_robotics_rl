# Implementation of Bipedal Walking with A2C - Advantage Actor Critics
# By Minh Nguyen
# ECE 5984 - Reinforcement Learning
# 11/21/2021

# Inspired Reference: https://github.com/lnpalmer/A2C
# Inspired Reference: https://github.com/floodsung/a2c_cartpole_pytorch
# Reading: https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html


import gym
import torch
import os
import sys
import pickle
import time
from os import path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.utils.additional_torch import to_device
from algorithms.a2c_algorithm import Policy
from algorithms.a2c_algorithm import Value
from algorithms.a2c_algorithm import a2c_step
from algorithms.a2c_algorithm import estimate_advantages
from algorithms.a2c_algorithm import BipedalWalkerAgent
from algorithms.utils import ZFilter
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# initialize global variable
L2_REG = 1e-3
GAMMA = 0.99
TAU = 0.95
MAX_NUM_ITER = 1000  # 50000
RENDER = False
MIN_BATCH_SIZE = 2048
LOG_INTERVAL = 1
SAVE_MODEL_INTERVAL = 2
ENV_NAME = "BipedalWalker-v2"

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
running_state = ZFilter((state_dim,), clip=5)

# seeding
np.random.seed(1)
torch.manual_seed(1)
env.seed(1)

# define actor and critic network
policy_net = Policy(state_dim, env.action_space.shape[0], log_std=-1.0)
value_net = Value(state_dim)
policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=4e-4)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=8e-4)

# create Bipedal Walker Agent
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


def update_a2c_params(batch):
    """Updates training parameters by taking steps from A2C algorithm

    Args:
        batch: input batch
    """
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)

    (
        advantages,
        returns,
    ) = estimate_advantages(  # get estimated advantage from stepping trajectories
        rewards, masks, values, GAMMA, TAU, device
    )

    a2c_step(  # run A2C algorithm updates
        policy_net,
        value_net,
        optimizer_policy,
        optimizer_value,
        states,
        actions,
        returns,
        advantages,
        L2_REG,
    )


def a2c_main():
    """User Interface"""
    t0 = time.time()

    # plot
    plot = plt.figure()
    xval, yval = [], []
    subplot = plot.add_subplot()
    plt.xlabel("Number Episodes")
    plt.ylabel("Rewards")
    plt.title("Rewards vs Number Episodes")
    (plotLine,) = subplot.plot(xval, yval)
    subplot.set_xlim([0, MAX_NUM_ITER])
    subplot.set_ylim([-400, 400])

    # run iteration
    for i_iter in range(MAX_NUM_ITER):
        # generates multiple trajectories that reach the min_batch_size
        batch, log = agent.collect_samples(MIN_BATCH_SIZE, RENDER)

        update_a2c_params(batch)

        if i_iter % LOG_INTERVAL == 0:
            print(f'Episode {i_iter+1} finished. Highest reward: {log["max_reward"]}')

        # plot
        xval.append(i_iter)
        yval.append(log["max_reward"])
        plotLine.set_xdata(xval)
        plotLine.set_ydata(yval)
        plot.savefig("./results/a2c_max_reward")

        # optional: save trained model
        if SAVE_MODEL_INTERVAL > 0 and (i_iter + 1) % SAVE_MODEL_INTERVAL == 0:
            to_device(torch.device("cpu"), policy_net, value_net)

            pickle.dump(  # write the trained model to folder
                (policy_net, value_net, running_state),
                open(
                    os.path.join(
                        saved_assets_dir(),
                        "learned_models/a2c_algorithm/bipedal_walker_v2_a2c.p",
                    ),
                    "wb",
                ),
            )
            to_device(device, policy_net, value_net)

        # clean up gpu memory after every iteration
        torch.cuda.empty_cache()

    t1 = time.time()
    print(f"All episodes finished. Training time of A2C is: {t1-t0}")
