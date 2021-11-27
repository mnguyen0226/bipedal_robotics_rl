# Implementation of Bipedal Walking with A2C - Advantage Actor Critics
# By Minh Nguyen
# ECE 5984 - Reinforcement Learning
# 11/21/2021

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
from algorithms.a2c_algorithm import Agent
from algorithms.utils import ZFilter
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################################
l2_reg = 1e-3
gamma = 0.99
tau = 0.95
max_num_iter = 1000  # 50000
render = False
min_batch_size = 2048
log_interval = 1
save_model_interval = 2
env_name = "BipedalWalker-v2"

###############################
dtype = torch.float64
torch.set_default_dtype(dtype)

# set device
device = (
    torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")
)
if torch.cuda.is_available():
    torch.cuda.set_device(0)

# environment
env = gym.make(env_name)
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

# create agent
agent = Agent(
    env,
    policy_net,
    device,
    running_state=running_state,
    render=False,
    num_threads=1,
)


def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), "../assets"))


def update_params(batch):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(
        rewards, masks, values, gamma, tau, device
    )

    """perform mini batch A2C update"""
    a2c_step(
        policy_net,
        value_net,
        optimizer_policy,
        optimizer_value,
        states,
        actions,
        returns,
        advantages,
        l2_reg,
    )


###############################
def a2c_main():
    # plot
    plot = plt.figure()
    xval, yval = [], []
    subplot = plot.add_subplot()
    plt.xlabel("Number Episodes")
    plt.ylabel("Rewards")
    plt.title("Rewards vs Number Episodes")
    (plotLine,) = subplot.plot(xval, yval)
    subplot.set_xlim([0, max_num_iter])
    subplot.set_ylim([-400, 400])

    # run iteration
    for i_iter in range(max_num_iter):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(min_batch_size, render)

        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        if i_iter % log_interval == 0:
            print(
                "{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}".format(
                    i_iter,
                    log["sample_time"],
                    t1 - t0,
                    log["min_reward"],
                    log["max_reward"],
                    log["avg_reward"],
                )
            )

        # plot
        xval.append(i_iter)
        yval.append(log["max_reward"])
        plotLine.set_xdata(xval)
        plotLine.set_ydata(yval)
        plot.savefig("./results/a2c_max_reward")

        if save_model_interval > 0 and (i_iter + 1) % save_model_interval == 0:
            to_device(torch.device("cpu"), policy_net, value_net)

            pickle.dump(
                (policy_net, value_net, running_state),
                open(
                    os.path.join(
                        assets_dir(),
                        "learned_models/a2c_algorithm/bipedal_walker_v2_a2c.p",
                    ),
                    "wb",
                ),
            )
            to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()

    print("All episodes finished.")


if __name__ == "__main__":
    a2c_main()
